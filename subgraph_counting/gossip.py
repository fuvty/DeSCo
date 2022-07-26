import csv
import math
import os
import random
import sys
from collections import defaultdict
from ctypes.wintypes import INT
from typing import List, Set, Tuple

import graph_tool.all as gt
import torch_geometric as pyg
from networkx.generators import directed
from numpy.core.fromnumeric import mean

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from common import data, models, utils
from subgraph_matching.config import parse_encoder
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from subgraph_counting.canonical_analyze import (analyze_graphlet_counts,
                                                 analyze_neighborhood_counts)
from subgraph_counting.config import parse_count, parse_gossip
from subgraph_counting.data import (OTFSynCanonicalDataSource, count_canonical,
                                    count_canonical_mp, count_graphlet,
                                    get_neigh_canonical, load_data,
                                    sample_graphlet, sample_neigh_canonical,
                                    true_count_anchor)
from subgraph_counting.models import (BaseLineModel, CanonicalCountModel,
                                      GossipModel, MotifCountModel,
                                      MultiTaskModel)
from subgraph_counting.train import get_workload
from subgraph_counting.transforms import (NetworkxToHetero, ToTCONV,
                                          ToTconvHetero)
from subgraph_counting.workload import Workload, graph_atlas_plus

BATCH_SIZE = 64

HETERO = True
TCONV_T = True
TCONV_Q = True

def train_gossip(model: GossipModel, query_ids: List[INT], neighs_pyg, args, neighs_valid= None, device='cuda', **kwargs):
    '''
    if use PFCONV, query_embs(dict[query_id]= pyg.Graph) should be passed to kwargs
    args: lr, weight_decay, dataset, model_path, num_epoch
    '''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    neighs_pyg = [b.to(device) for b in neighs_pyg]
    # all data needed is prepared

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # add schedular
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    model.train()
    print('training model')
    name = args.dataset + "_" + "gossip" + "_" + "n_query_"+str(len(query_ids)) + "_" + "all"
    logger = SummaryWriter(comment=name+ "_" + args.model_path)

    for epoch in tqdm(range(args.num_epoch)):
        acc_loss = []
        for batch in DataLoader(neighs_pyg, batch_size=64):
            optimizer.zero_grad()

            loss_queries = []
            for query_id in query_ids:
                batch.node_feature = batch["eval_"+str(query_id)]
                if model.emb_with_query:
                    query_emb= kwargs["query_embs"][query_id].detach() # do not update query emb here
                else:
                    query_emb= None
                pred_counts = model.gossip_count(batch, query_emb= query_emb)
                pred_counts = pred_counts+batch["eval_"+str(query_id)][:,0].view(-1,1)
                loss = model.criterion(pred_counts, batch["count_"+str(query_id)]) # pred diff
                acc_loss.append(loss)

                loss_queries.append(loss)
            
            loss = torch.sum(torch.stack(loss_queries))
            loss.backward()
            optimizer.step()
            # print("Epoch {}/{}. Batch {}/{}. Loss: {:.4f}."
            # .format(epoch, args.num_epoch, b, len(neighs_batch_train), loss), end="                                                    \r")
            
        logger.add_scalar("Loss/train", mean([n.cpu().item() for n in acc_loss]), epoch)

        torch.save({'state_dict': model.state_dict(), 'args': model.args}, args.model_path)
        scheduler.step(torch.mean(torch.stack(acc_loss)))

        if neighs_valid is not None:
            model.eval()
            acc_loss_valid = []
            
            for batch in DataLoader(neighs_valid, batch_size=64):
                for query_id in query_ids:
                    batch.node_feature = batch["eval_"+str(query_id)]
                    if model.emb_with_query:
                        query_emb= kwargs["query_embs"][query_id].detach()
                    else:
                        query_emb= None
                    pred_counts = model.gossip_count(batch, query_emb= query_emb)
                    pred_counts = pred_counts+batch["eval_"+str(query_id)][:,0].view(-1,1)
                    loss = model.criterion(pred_counts, batch["count_"+str(query_id)]) # pred diff
                    acc_loss_valid.append(loss)

            acc_loss_valid_mean = torch.mean(torch.stack(acc_loss_valid))
            print("Epoch {}/{}. Valid Loss: {:.4f}. ".format(epoch, args.num_epoch, acc_loss_valid_mean), end="             \n")

            logger.add_scalar("Loss/valid", acc_loss_valid_mean, epoch)
            model.train()

    torch.save({'state_dict': model.state_dict(), 'args': model.args}, args.model_path)
    
    logger.flush()
    logger.close()

    print(args)
    return 0

def evaluate_gossip(model: GossipModel, query_ids: List[INT], neighs_pyg, args, device='cuda', **kwargs):
    '''
    reture tuple of tensors pred_counts, canonical_counts, truth_counts, each of shape (#query, #neighborhood)
    '''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    neighs_pyg = [b.to(device) for b in neighs_pyg]
    # all data needed is prepared

    model.eval()
    model = model.to(device)
    pred_counts = []
    truth_counts = []
    pred_counts_tensor = [] # len = len(query_ids)
    canonical_counts_tensor = []
    truth_counts_tensor = []

    start_time = time.time()
    with torch.no_grad():
        for query_id in tqdm(query_ids): # ALERT: use only one query
            pred_counts = []
            truth_counts = []
            canonical_counts = []
            for batch in DataLoader(neighs_pyg, batch_size= 64):
                batch.node_feature = batch["eval_"+str(query_id)]
                if model.emb_with_query:
                    query_emb= kwargs["query_embs"][query_id].detach() # do not update query emb here
                else:
                    query_emb= None

                pred_count = model.gossip_count(batch, query_emb= query_emb) + batch["eval_"+str(query_id)][:,0].view(-1,1)
                truth_count = batch["count_"+str(query_id)].to(device)
                canonical_count = batch["eval_"+str(query_id)][:,0].view(-1,1).to(device)

                if args.use_log:
                    truth_count = 2**truth_count - 1
                    pred_count = F.relu(2**pred_count - 1)
                    canonical_count = F.relu(2**canonical_count - 1)

                pred_count_graphlet = pyg.nn.global_add_pool(pred_count, batch.batch).view(-1)
                truth_counts_graphlet = pyg.nn.global_add_pool(truth_count, batch.batch).view(-1)
                canonical_counts_graphlet = pyg.nn.global_add_pool(canonical_count, batch.batch).view(-1)

                pred_counts.append(pred_count_graphlet)
                truth_counts.append(truth_counts_graphlet)
                canonical_counts.append(canonical_counts_graphlet)
    
            pred_counts = torch.cat(pred_counts, dim=0)
            truth_counts =  torch.cat(truth_counts, dim=0)
            canonical_counts = torch.cat(canonical_counts, dim=0)
            # graphlet count of shape [#graphs]

            pred_counts = torch.round(pred_counts)
            canonical_counts = torch.round(canonical_counts)

            pred_counts_tensor.append(pred_counts.view(-1))
            truth_counts_tensor.append(truth_counts.view(-1))
            canonical_counts_tensor.append(canonical_counts.view(-1))
        
        pred_counts_tensor = torch.stack(pred_counts_tensor, dim=0)
        canonical_counts_tensor = torch.stack(canonical_counts_tensor, dim=0)
        truth_counts_tensor = torch.stack(truth_counts_tensor, dim=0)

    end_time = time.time()
    print("Gossip Counting time: {:.4f}".format(end_time-start_time))

    return pred_counts_tensor, canonical_counts_tensor, truth_counts_tensor


def CanonicalInference(model, query_ids: List[int], neighs_pyg, device='cpu', mode= 'motif') -> List[torch.Tensor]:
    '''
    input:
    output: the inference counts of all the batches
    '''
    model = model.to(device)

    # begin {query}, commonly used for all
    # convert nx_graph queries to pyg
    query_embs = dict()
    if HETERO:
        queries_pyg = [NetworkxToHetero(graph_atlas_plus(query_id), type_key= 'type', feat_key= 'feat').to(device) for query_id in query_ids]
    else:
        queries_pyg = [pyg.utils.from_networkx(graph_atlas_plus(query_id)).to(device) for query_id in query_ids]
    for query_pyg in queries_pyg:
        if HETERO:
            query_pyg['union_node'].node_feature = torch.zeros((query_pyg['union_node'].num_nodes, 1), device= device)
        else:
            query_pyg.node_feature = torch.zeros((query_pyg.num_nodes, 1), device= device)
    # end {query}, commonly used for all

    # commonly used for all
    # prepare dataset and workload
    neighs_train = neighs_pyg # used for training

    start_time = time.time()
    if TCONV_T:
        # transform = T.Compose([ToTCONV(node_type='count', node_attr='node_feature')])
        transform = T.Compose([ToTconvHetero(node_attr='node_feature')])

        neighs_cp = []
        for g in neighs_train:
            neighs_cp.append(transform(g))
        neighs_train = neighs_cp

    if TCONV_Q:
        # transform = T.Compose([ToTCONV(node_type='union_node', node_attr='node_feature')])
        transform = T.Compose([ToTconvHetero(node_attr='node_feature')])

        neighs_cp = []
        for g in queries_pyg:
            neighs_cp.append(transform(g))
        queries_pyg = neighs_cp
    end_time = time.time()
    if TCONV_T or TCONV_Q:
        print("subgraph-based heterogenegous converlution transform time: {:.4f}".format(end_time - start_time))

    query_loader = DataLoader(queries_pyg, batch_size= 64)
    neighs_loader_train = DataLoader(neighs_train, batch_size= BATCH_SIZE)

    start_time = time.time()
    emb_queries = []
    for query_batch in query_loader:
        emb_queries.append(model.emb_model_query(query_batch))
    emb_queries = torch.cat(emb_queries, dim=0)

    count_query = [[] for q in range(len(query_ids))]
    for batch_pyg in tqdm(neighs_loader_train):
        emb_target = model.emb_model(batch_pyg.to(device))
        
        for i, query_id in enumerate(query_ids):
            emb_query = emb_queries[i].expand_as(emb_target)
            emb = (emb_target, emb_query)

            if mode == "motif":
                results = model.forward(emb)
                results = torch.clamp(results, min=0, max=16.0)
            elif mode == "multitask":
                results = model.forward(emb) 
                count = results['counting']
                count = torch.clamp(count, min=0, max=16.0)
                label = torch.floor(F.relu(count)).view(-1).long() # assuming log setting
                certainty = F.softmax(results['classification'], dim=1)
                num_class = certainty.shape[-1]
                label, _ = torch.min(torch.stack((label, (num_class-1)*torch.ones_like(label)), dim=1), dim=1) # label is no larger than number of classes
                index = torch.tensor([i for i in range(num_class)], device= certainty.device).expand_as(certainty) == label.unsqueeze(1).expand(-1, num_class)
                certainty = certainty[index].view(-1,1)
                results = torch.cat((count, certainty), dim=-1)
            
            count_query[i].append(results)
    end_time = time.time()    
    print('Neighborhood Counting time: {:.4f}'.format(end_time - start_time))
    
    return [torch.cat(count_query[query_id], dim=0) for query_id in range(len(query_ids))], emb_queries # List[Tensor(shape==node*feature_size)]

def update_nx_node_eval(nx_targets: List[nx.Graph], query_ids, input_dim, device):
    for g in nx_targets:
        for n in g.nodes:
            for query_id in query_ids:
                if input_dim == 1:
                    g.nodes[n]["eval_"+str(query_id)] = torch.tensor([0.0], device= device)
                else:
                    g.nodes[n]["eval_"+str(query_id)] = torch.cat( (torch.tensor([0.0], device= device), torch.ones(input_dim-1, device= device)), dim=0)

def get_canonical_model(model_args, model_path):
    model = BaseLineModel(1, 64, model_args)
    # model = MultiTaskModel(1, 64, model_args, task=['counting', 'classification'])
    # basemodel = torch.load("ckpt/sage_345_syn_epo900.pt")
    # model = sage2multitask(basemodel, model, args)
    if HETERO:
        if TCONV_T:
            model.emb_model.gnn_core = pyg.nn.to_hetero(model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union_triangle', 'count'), ('count', 'union_tride', 'count'), ('count', 'union_triangle', 'canonical'), ('count', 'union_tride', 'canonical'), ('canonical', 'union_triangle', 'count'), ('canonical', 'union_tride', 'count')] ), aggr='sum')
        else:
            model.emb_model.gnn_core = pyg.nn.to_hetero(model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union', 'canonical'), ('canonical', 'union', 'count'), ('count', 'union', 'count')] ), aggr='sum')
        
        if TCONV_Q:
            model.emb_model_query.gnn_core = pyg.nn.to_hetero(model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union_triangle', 'union_node'), ('union_node', 'union_tride', 'union_node')] ), aggr='sum')
        else:
            model.emb_model_query.gnn_core = pyg.nn.to_hetero(model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union', 'union_node')] ), aggr='sum')
    
    model.load_state_dict(torch.load(model_path)['state_dict'])

    return model

def main(train= False, test= False, force_dataset= None, device= 'cuda', neighborhood_counting_model_path= 'ckpt/general/trans/tconv/sage_main_model.pt', quick_return_gossip_graph= False, valid_gossip_graph= None):

    # gen query_ids
    atlas_graph = defaultdict(list)
    for i in range(4, 1253):
    # for i in range(4,53):
        g = graph_atlas_plus(i) # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_graph[len(g)].append(i)
    query_ids = atlas_graph[3] + atlas_graph[4] + atlas_graph[5] # + atlas_graph[6] + atlas_graph[7]
    # query_ids = [81, 103, 276, 320, 8006, 8007, 9006, 9007, 10006, 10007, 11006, 11007, 12006, 12007, 13006, 13007]

    print("num_queries",len(query_ids))

    # define gossip model
    # make model
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    parse_gossip(parser)
    args_gossip = parser.parse_args() 
    input_dim = 1
    args_gossip.use_hetero = False
    model = GossipModel(input_dim, 64, args_gossip, emb_channels= 64, input_pattern_emb= True).to(device) # assume emb channel is 64
    # model = GossipModel(1, 64, args, emb_channels= 64).to(device) # assume emb channel is 64
    model.emb_with_query = True

    # load args for neighborhood counting
    parser = argparse.ArgumentParser(description='Canonical Count')
    utils.parse_optimizer(parser)
    parse_count(parser)
    neigh_count_args = parser.parse_args()
    neigh_count_args.hetero = HETERO

    # define workload
    # if inference, set data to enzymes
    if force_dataset is not None:
        neigh_count_args.dataset = force_dataset
    args_gossip.dataset = neigh_count_args.dataset

    # define neighborhood counting model
    print("canonical inference with model", neighborhood_counting_model_path)
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args_canonical_model = parser.parse_args() 
    args_canonical_model.use_hetero = HETERO
    canonical_model = get_canonical_model(args_canonical_model, neighborhood_counting_model_path).to(device)

    # begin neighborhood counting
    workload = get_workload(query_ids, neigh_count_args, load_list=['neighs_pyg', 'graphs_nx'])
    neighs_pyg = workload.neighs_pyg
    print("load neighs_pyg", len(neighs_pyg))
    with torch.no_grad():
        inference_results, emb_queries = CanonicalInference(canonical_model, query_ids, neighs_pyg, device= device, mode= 'motif')

    # use inference results as inputs of the gossip model
    print("convert to gossip graph")
    nx_targets = workload.graphs_nx
    update_nx_node_eval(nx_targets, query_ids, input_dim, device)
    neighs_index = workload.neighs_index
    for query_id, inference_result in tzip(query_ids ,inference_results):
        for k,result in enumerate(inference_result):
            gid,node = neighs_index[k]
            nx_targets[gid].nodes[node]["eval_"+str(query_id)] = result
    
    # convert nx_targets to pyg_batch
    with torch.no_grad():
        pyg_targets = []
        for g in tqdm(nx_targets):
            g = pyg.utils.from_networkx(g)
            for key in ["eval_"+str(qid) for qid in query_ids] + ["count_"+str(qid) for qid in query_ids]:
                if type(g[key]) is list:
                    g[key] = torch.stack(g[key], dim=0)
                elif type(g[key]) is torch.Tensor:
                    g[key] = g[key].unsqueeze(dim=-1)
                else:
                    raise NotImplementedError
            pyg_targets.append(g)
        gossip_graph = pyg_targets

    if quick_return_gossip_graph:
        return gossip_graph

    # train gossip model
    if train:
        if model.emb_with_query:
            query_embs = dict()
            queries_pyg = [pyg.utils.from_networkx(graph_atlas_plus(query_id)).to(device) for query_id in query_ids]
            for query_pyg in queries_pyg:
                query_pyg.node_feature = torch.zeros((query_pyg.num_nodes, 1), device= device)
            for i, query_id in enumerate(query_ids):
                query_embs[query_id] = emb_queries[i]
            train_gossip(model, query_ids, gossip_graph, args_gossip, device=device, query_embs= query_embs, neighs_valid= valid_gossip_graph)
        else:
            train_gossip(model, query_ids, gossip_graph, args_gossip, device=device, neighs_valid= valid_gossip_graph)
        print("gossip model ", args_gossip.model_path)
    
    # test gossip model
    if test:
        # assert(args.dataset=="ENZYMES")
        # model_name = "ckpt/gossip/sage_345_syn6400_gossip_epo300.pt"
        print("evaluate with model", args_gossip.model_path)
        model_name = args_gossip.model_path
        # model = torch.load(model_name).to(device)
        model.load_state_dict(torch.load(model_name)['state_dict'])

        if model.emb_with_query:
            query_embs = dict()
            queries_pyg = [pyg.utils.from_networkx(graph_atlas_plus(query_id)).to(device) for query_id in query_ids]
            for query_pyg in queries_pyg:
                query_pyg.node_feature = torch.zeros((query_pyg.num_nodes, 1), device= device)
            for i, query_id in enumerate(query_ids):
                query_embs[query_id] = emb_queries[i]
            pred_counts, canonical_counts, truth_counts = evaluate_gossip(model, query_ids, gossip_graph, args_gossip, device=device, query_embs= query_embs)
        else:
            pred_counts, canonical_counts, truth_counts = evaluate_gossip(model, query_ids, gossip_graph, args_gossip, device=device)

        print("ground_truth", truth_counts)
        print("canonical_count", canonical_counts)
        print("gossip_count", pred_counts)

        analyze_graphlet_counts(pred_counts, truth_counts, query_ids, model_name= '/'.join(model_name.split('.')[0].split('/')[1:]), dataset=args_gossip.dataset, device= device, compute_confusion_matrix= False)

    return gossip_graph

if __name__ == '__main__':
    # gossip_graph_valid = main(train= False, test= True, force_dataset= "ENZYMES", quick_return_gossip_graph= True) # valid dataset
    # main(train= True ,test= False, valid_gossip_graph= gossip_graph_valid) # train
    main(train= False, test= True, force_dataset= "ENZYMES") # test
    # main(train= False, test= True, force_dataset= "COX2") # test
    # main(train= False, test= True, force_dataset= "MUTAG") # test
    
    # main(train= False, test= True, force_dataset= "CiteSeer") # test
    # main(train= False, test= True, force_dataset= "Cora") # test
