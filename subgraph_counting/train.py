"""Train the regression subgraph counting model"""
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict
from typing import List
import graph_tool.all as gt

import torch_geometric as pyg
from numpy.core.fromnumeric import mean

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import multiprocessing as mp
import pickle
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from common import data, models, utils
from subgraph_matching.config import parse_encoder
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from tqdm import tqdm

from subgraph_counting.config import parse_count
from subgraph_counting.models import (BaseLineModel, CanonicalCountModel,
                                      MotifCountModel, MultiTaskModel)
from subgraph_counting.canonical_analyze import eval_canonical
from subgraph_counting.workload import Workload, graph_atlas_plus
from subgraph_counting.transforms import ToTCONV, NetworkxToHetero, ToTconvHetero, to_device, get_truth
from subgraph_counting.baseline.GNNSubstructure.LRP_dataset import LRP_Dataset, collate_lrp_dgl_light_index_form_wrapper

SAVE_EVERY_EPOCH = 50
BATCH_SIZE = 64
GRAPHLET = False
FINETUNE = False
LRP = False
HETERO = True
TCONV_T = True
TCONV_Q = True

def train_canonical(model, query_ids: list[int], neighs_pyg_dict: dict[str, pyg.data.dataset.Dataset], device, optimizer, args):
    '''
    valid suggests whether you want to use validation data for training the subgraph model.
    '''
    print('train with args')
    print(args)
    # training settings
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001) # add schedular

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

    # prepare dataset and workload
    neighs_train = neighs_pyg_dict['train'] # used for training
    neighs_valid = neighs_pyg_dict['valid'] # used for validation
    if TCONV_Q:
        # transform = T.Compose([ToTCONV(node_type='union_node', node_attr='node_feature')])
        transform = T.Compose([ToTconvHetero(node_attr='node_feature')])

        neighs_cp = []
        for g in queries_pyg:
            neighs_cp.append(transform(g))
        queries_pyg = neighs_cp

    if TCONV_T:
        # transform = T.Compose([ToTCONV(node_type='count', node_attr='node_feature')])
        transform = T.Compose([ToTconvHetero(node_attr='node_feature')])

        neighs_cp = []
        for g in neighs_train:
            neighs_cp.append(transform(g))
        neighs_train = neighs_cp

        neighs_cp = []
        for g in neighs_valid:
            neighs_cp.append(transform(g))
        neighs_valid = neighs_cp

    
    if LRP:
        # get query LRP
        queries_pyg = [g.to('cpu') for g in queries_pyg]
        query_LRP = LRP_Dataset('queries_'+str(len(queries_pyg)), graphs= queries_pyg, labels= [torch.zeros(1) for _ in range(len(queries_pyg))], lrp_save_path= '/home/nfs_data/futy/repos/prime/GNN_Mining/2021Summer/tmp_folder', lrp_depth=1, subtensor_length=4, lrp_width=3)

        query_loader = torch_data.DataLoader(query_LRP, batch_size=BATCH_SIZE, shuffle= False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(4))

        neighs_loader_train = torch_data.DataLoader(neighs_train, batch_size= BATCH_SIZE, shuffle=False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(4))

        neighs_loader_valid = torch_data.DataLoader(neighs_train, batch_size= BATCH_SIZE, shuffle=False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(4))
    else:
        query_loader = DataLoader(queries_pyg, batch_size= BATCH_SIZE)
        neighs_loader_train = DataLoader(neighs_train, batch_size= BATCH_SIZE)
        neighs_loader_valid = DataLoader(neighs_valid, batch_size= BATCH_SIZE)
    
    print("******************")
    print(args)

    # begin training and validation
    print('training model')
    logger = SummaryWriter(comment=args.model_path)

    # make dir for saving model if not exist
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    for epoch in range(args.num_epoch):
        model.train() # switch to training mode
        acc_loss = []
        for batch_i, batch_pyg in enumerate(neighs_loader_train):
            optimizer.zero_grad() # exclusive for training, not valid
            # begin {inference}, commonly used for all
            batch_pyg = to_device(batch_pyg, device)
            if model.emb_with_query:
                raise NotImplementedError
            elif hasattr(model, 'DIAMNet'):
                emb_queries = []
                query_lens = []
                for query_batch in query_loader:
                    emb, query_len = model.emb_model_query(query_batch)
                    emb_queries.append(emb)
                    query_lens.append(query_len)
                emb_queries = torch.cat(emb_queries, dim=0)
                query_lens = torch.cat(query_lens, dim=0)
                emb_queries = [(emb_queries[i], query_lens[i]) for i in range(emb_queries.shape[0])] # List[Tuple[Tensor, Tensor]], each represent a graph
                emb_target = model.emb_model(batch_pyg)
            else:
                emb_queries = []
                for query_batch in query_loader:
                    query_batch = to_device(query_batch, device)
                    emb_queries.append(model.emb_model_query(query_batch))
                emb_queries = torch.cat(emb_queries, dim=0)
                emb_target = model.emb_model(batch_pyg)
            
            for i, query_id in enumerate(query_ids):
                query_embs[query_id] = emb_queries[i]
            loss_queries = []
            for i, query_id in enumerate(query_ids):
                if hasattr(model, 'DIAMNet'):
                    query_emb = (
                        query_embs[query_id][0].repeat(emb_target[0].shape[0], 1, 1),
                        query_embs[query_id][1].repeat(emb_target[1].shape[0], 1),
                    )
                else:
                    query_emb = query_embs[query_id].expand_as(emb_target)
                
                emb = (emb_target, query_emb)
                results = model.forward(emb)

                truth = get_truth(batch_pyg)[:,i].view(-1,1) # shape (batch_size,1)
                loss = model.criterion(results, truth)
                loss_queries.append(loss)
                acc_loss.append(loss.item())       
            # end {inference}, commonly used for all
            loss = torch.mean(torch.stack(loss_queries))
            loss.backward()
            optimizer.step()

            print("Epoch {}/{}. Batch {}/{}. Loss: {:.4f}.".format(epoch, args.num_epoch, batch_i, len(neighs_loader_train), loss), end="                                                    \r")


        # record in tensorboard
        logger.add_scalar("Loss/train", mean(acc_loss), epoch)

        # valid
        model.eval() # switch to evaluation mode
        acc_loss = []
        with torch.no_grad():
            for batch_i, batch_pyg in enumerate(neighs_loader_valid):
                # begin {inference}, commonly used for all
                batch_pyg = to_device(batch_pyg, device)
                if model.emb_with_query:
                    raise NotImplementedError
                elif hasattr(model, 'DIAMNet'):
                    emb_queries = []
                    query_lens = []
                    for query_batch in query_loader:
                        emb, query_len = model.emb_model_query(query_batch)
                        emb_queries.append(emb)
                        query_lens.append(query_len)
                    emb_queries = torch.cat(emb_queries, dim=0)
                    query_lens = torch.cat(query_lens, dim=0)
                    emb_queries = [(emb_queries[i], query_lens[i]) for i in range(emb_queries.shape[0])] # List[Tuple[Tensor, Tensor]], each represent a graph
                    emb_target = model.emb_model(batch_pyg)
                else:
                    emb_queries = []
                    for query_batch in query_loader:
                        query_batch = to_device(query_batch, device)
                        emb_queries.append(model.emb_model_query(query_batch))
                    emb_queries = torch.cat(emb_queries, dim=0)
                    emb_target = model.emb_model(batch_pyg)
                
                for i, query_id in enumerate(query_ids):
                    query_embs[query_id] = emb_queries[i]
                loss_queries = []
                for i, query_id in enumerate(query_ids):
                    if hasattr(model, 'DIAMNet'):
                        query_emb = (
                            query_embs[query_id][0].repeat(emb_target[0].shape[0], 1, 1),
                            query_embs[query_id][1].repeat(emb_target[1].shape[0], 1),
                        )
                    else:
                        query_emb = query_embs[query_id].expand_as(emb_target)
                    
                    emb = (emb_target, query_emb)
                    results = model.forward(emb)
                    
                    truth = get_truth(batch_pyg)[:,i].view(-1,1) # shape (batch_size,1)
                    loss = model.criterion(results, truth)
                    loss_queries.append(loss)
                    acc_loss.append(loss.item())       
                # end {inference}, commonly used for all
            loss_valid = torch.sum(torch.stack(loss_queries))
            print("Epoch {}/{}. Valid Loss: {:.4f}. ".format(epoch, args.num_epoch, mean(acc_loss)), end="             \n")
            # record in tensorboard
            logger.add_scalar("Loss/valid", mean(acc_loss), epoch)

        # save model and checkpoints
        # torch.save(model.state_dict(), args.model_path)
        scheduler.step(loss_valid)
        if (epoch+1)%SAVE_EVERY_EPOCH==0: # save checkpoint every 100 epoch
            torch.save({'state_dict': model.state_dict(), 'args': model.args}, args.model_path+"_epo"+str(epoch+1)+".pt")
    # torch.save(model, args.model_path)
    
    logger.flush()
    logger.close()
    print(args)
    return 0

def build_model(args):
    # build model
    if args.method_type == "order":
        model = models.OrderEmbedder(1, args.hidden_dim, args)

def sage2tconv(sage_model, tconv_model, args):
    '''
    init params of tconv using the params from sage
    '''
    # tconv_model = MultiTaskModel(1, 64, args)
    sage_state_dict = sage_model.state_dict()
    tconv_state_dict = dict()
    for key,value in sage_state_dict.items():
        cmps = key.split('.')
        if cmps[1] == 'convs' and cmps[3] != 'lin_update':
            cmp_dns = cmps[3]+'_dns'
            cmp_sps = cmps[3]+'_sps'
            tconv_state_dict[".".join(cmps[0:3]+[cmp_dns]+cmps[4:])] = value 
            tconv_state_dict[".".join(cmps[0:3]+[cmp_sps]+cmps[4:])] = value 
        else:
            tconv_state_dict[key] = value
    tconv_model.load_state_dict(tconv_state_dict, strict=False)
    return tconv_model

def sage2tconv_hetero(sage_model, tconv_model):
    '''
    init params of tconv using the params from sage
    both models are trained under hetero setting
    '''
    sage_state_dict = sage_model.state_dict()
    tconv_state_dict = dict()
    for key,value in sage_state_dict.items():
        cmps = key.split('.')
        if cmps[1] == 'gnn_core' and cmps[2] == 'convs':
            if cmps[6] == 'bias':
                v = value/2
            else:
                v = value
            edge_type = cmps[4].split('__')
            cmp_triangle = '__'.join((edge_type[0],'union_triangle',edge_type[2]))
            cmp_tride = '__'.join((edge_type[0],'union_tride',edge_type[2]))
            tconv_state_dict[".".join(cmps[0:4]+[cmp_triangle]+cmps[5:])] = v 
            tconv_state_dict[".".join(cmps[0:4]+[cmp_tride]+cmps[5:])] = v 
        else:
            tconv_state_dict[key] = value
    assert(set(tconv_state_dict.keys()) == set(tconv_model.state_dict().keys()))
    tconv_model.load_state_dict(tconv_state_dict)
    return tconv_model

def get_PNA_deg(dataset, max_degree=30):
    deg = torch.zeros(max_degree, dtype=torch.long)
    ds = []
    for data in dataset:
        d = pyg.utils.degree(data.edge_index[1], num_nodes=data.node_feature.shape[0], dtype=torch.long).cpu()
        deg += torch.bincount(d, minlength=deg.numel())
        # ds.append(max(d))
    # print(max(ds))
    return deg

def get_workload_name(dataset, len_query_ids, n_neighborhoods=6400, use_norm=False, objective="canonical", relabel_mode=None, hetero= True):
    if dataset == 'syn':
        name = dataset + "_" + "gossip_" + str(n_neighborhoods) + "_" + "n_query_"+str(len_query_ids) + "_" + "all"
    else:
        name = dataset + "_" + "gossip_" + "n_query_"+str(len_query_ids) + "_" + "all"
    # if args.use_log:
    #     name += "_log"
    if use_norm:
        name += "_norm"
    if objective == "graphlet":
        name += "_graphlet"
    if relabel_mode is not None:
        name += "_"+relabel_mode
    name = name.replace("/", "_")

    if hetero:
        name = name + "_hetero"

    workload_file = "subgraph_counting/workload/general/" + name

    return workload_file

def get_workload(query_ids, args, load_list=['neighs_pyg']):
    '''
    args contains dataset, use_norm, objective, relabel_mode, n_neighborhoods, hetero
    '''
    if GRAPHLET:
        args.hetero = True
        # load_list.append('graphs_nx') if 'graphs_nx' not in load_list else None
        load_list = ['graphs_nx']

    if args.dataset == 'syn':
        name = args.dataset + "_" + "gossip_" + str(args.n_neighborhoods) + "_" + "n_query_"+str(len(query_ids)) + "_" + "all"
    else:
        name = args.dataset + "_" + "gossip_" + "n_query_"+str(len(query_ids)) + "_" + "all"
    # if args.use_log:
    #     name += "_log"
    if args.use_norm:
        name += "_norm"
    if args.objective == "graphlet":
        name += "_graphlet"
    if args.relabel_mode is not None:
        name += "_"+args.relabel_mode
    name = name.replace("/", "_")

    if args.hetero:
        name = name + "_hetero"

    workload_file = "subgraph_counting/workload/general/" + name
    workload_file_full = "subgraph_counting/workload/general/" + args.dataset + "_gossip_n_query_994_all" + "_hetero"

    if os.path.exists(workload_file):
        print("load ground truth from "+workload_file)
        workload = Workload(name, sample_neigh= False, hetero_graph= args.hetero)
        workload.load(workload_file, load_list= load_list)
    # elif os.path.exists(workload_file_full):
    #     print("load ground truth from full workload "+workload_file_full)
    #     workload = Workload(name, sample_neigh= False, hetero_graph= args.hetero)
    #     workload.load(workload_file_full, load_list= load_list)
    else:
        workload = Workload(name, sample_neigh= False, hetero_graph= args.hetero)
        print("generate and save ground truth to "+workload_file)
        workload.gen_workload_general(query_ids, args)
        workload.save(workload_file)

    if GRAPHLET:
        args.use_hetero = False
        print('convert to graphlet')
        workload.neighborhood_dataset = []
        for nx_graph in workload.graphs_nx:
            count_dict = dict()
            count_list = []
            # convert networkx to pyg and init node feature with zero tensor
            pyg_graph = pyg.utils.from_networkx(nx_graph)
            pyg_graph.node_feature = torch.zeros(len(nx_graph), 1)
            for key in pyg_graph.keys:
                if key.split('_')[0] == 'count':
                    query_id = int(key.split('_')[1])
                    count_dict[query_id] = torch.log2(torch.sum(2**pyg_graph[key]-1)+1)
            query_ids = sorted(list(count_dict.keys()))
            for query_id in query_ids:
                count_list.append(count_dict[query_id])
            pyg_graph.y = torch.stack(count_list, dim=0).view(1,-1)

            workload.neighborhood_dataset.append(pyg_graph)

    return workload

class DIAMNet_args:
    def __init__(self) -> None:
        self.hidden_dim = 128
        self.dropout = 0.0
        self.n_layers = 5
        # self.conv_type = 'RGIN'
        self.conv_type = 'GIN'
        self.use_hetero = False

class LRP_args:
    def __init__(self) -> None:
        self.hidden_dim = 8
        self.dropout = 0.0
        self.n_layers = 8
        self.use_hetero = False


def main(train= True, valid= True, test= True, test_dataset= "ENZYMES"):
    print('use config, ', 'graphlet ', GRAPHLET, 'hetero ', HETERO, 'TCONV_T ', TCONV_T, 'TCONV_Q', TCONV_Q)

    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)

    model_args = parser.parse_args()

    # define model
    # if args.conv_type == 'PNACONV':
    #     workload_file = "subgraph_counting/workload/syn_motif_n_query_29_n_neighs_57600_log"
    #     warnings.warn(('using deg from {}'.format(workload_file)))
    #     with open(workload_file, 'rb') as f:
    #         workload = pickle.load(f)
    #     deg = get_PNA_deg(workload.neighs_batch_train, max_degree=28) # 28 for syn_motif_n_query_29_n_neighs_57600_log
    #     model = BaseLineModel(1, 64, args, dataset=workload_file, deg=deg, towers=1)
    # else:
    #     model = BaseLineModel(1, 64, args)

    model_args.use_hetero = HETERO
    model = BaseLineModel(1, 64, model_args)

    # load and fine tune
    if FINETUNE:
        # load sage model
        sage_model_path = "ckpt/general/trans/sage_345_synXL_qs_hetero_update_epo250.pt"
        print('fine tune with model ', sage_model_path)
        sage_model = BaseLineModel(1, 64, model_args)
        sage_model.emb_model.gnn_core = pyg.nn.to_hetero(sage_model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union', 'canonical'), ('canonical', 'union', 'count'), ('count', 'union', 'count')] ), aggr='sum')
        sage_model.emb_model_query.gnn_core = pyg.nn.to_hetero(sage_model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union', 'union_node')] ) )
        sage_model.load_state_dict(torch.load(sage_model_path)['state_dict'])


    # diamnet_args = DIAMNet_args()
    # for key in diamnet_args.__dict__:
    #     setattr(model_args, key, getattr(diamnet_args, key))
    # model = BaseLineModel(1, 128, model_args, baseline= 'DIAMNet')
    # model.DIAMNet = True # add key DIAMNet in model keys

    # lrp_args = LRP_args()
    # for key in lrp_args.__dict__:
    #     setattr(model_args, key, getattr(lrp_args, key))   
    # model = BaseLineModel(1, lrp_args.hidden_dim, model_args, baseline= 'LRP')

    if HETERO:
        if TCONV_T:
            model.emb_model.gnn_core = pyg.nn.to_hetero(model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union_triangle', 'count'), ('count', 'union_tride', 'count'), ('count', 'union_triangle', 'canonical'), ('count', 'union_tride', 'canonical'), ('canonical', 'union_triangle', 'count'), ('canonical', 'union_tride', 'count')] ), aggr='sum')
        else:
            model.emb_model.gnn_core = pyg.nn.to_hetero(model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union', 'canonical'), ('canonical', 'union', 'count'), ('count', 'union', 'count')] ), aggr='sum')
        
        if TCONV_Q:
            model.emb_model_query.gnn_core = pyg.nn.to_hetero(model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union_triangle', 'union_node'), ('union_node', 'union_tride', 'union_node')] ), aggr='sum')
        else:
            model.emb_model_query.gnn_core = pyg.nn.to_hetero(model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union', 'union_node')] ), aggr='sum')

    ##### transfer
    # base_model_path = "ckpt/general/trans/tconv/sage_main_model.pt"
    # model.load_state_dict(torch.load(base_model_path)['state_dict'])
    ########

    # fine tune
    if FINETUNE:
        assert(TCONV_T and TCONV_Q)
        model = sage2tconv_hetero(sage_model, model) 

    print(model)
    print("model args")
    print(model_args)
 
    parser = argparse.ArgumentParser(description='Canonical Count')
    utils.parse_optimizer(parser)
    parse_count(parser)
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # define query
    # all 994 graphs with 3~7 nodes
    # 3~5 with 29 graphs
    # gen query_ids
    atlas_graph = defaultdict(list)
    for i in range(4, 1253):
    # for i in range(4,53):
        g = graph_atlas_plus(i) # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_graph[len(g)].append(i)
    query_ids = atlas_graph[3] + atlas_graph[4] + atlas_graph[5] # + atlas_graph[6] + atlas_graph[7]

    # atlas ids used for Large Queries:
    # size6: 83, 103; size7: 286, 320
    # query_ids += [83, 103, 286, 320, 8000, 8001, 9000, 9001, 10000, 10001, 11000, 11001, 12000, 12001, 13000, 13001, 14000, 14001]

    # query with maximum diameter 4
    # query_ids = [81, 103, 276, 320, 8006, 8007, 9006, 9007, 10006, 10007, 11006, 11007, 12006, 12007, 13006, 13007]

    print("num_queries",len(query_ids))
    '''
    # index of d-regular graph for graph_atlas
    # node 3~7 with number of graphs [1, 2, 2, 5, 4]
    queries = []
    d_regular_index = [7, 16,18, 38,52, 105,174,175,204,208, 353,1170,1171,1252]
    for i in  d_regular_index:
        queries.append(graph_atlas_plus(i))
    '''

    neighs_pyg_dict = dict()
    if train: 
        args.hetero = HETERO
        workload = get_workload(query_ids, args)
        neighs_pyg_dict['train'] = workload.neighs_pyg
        if LRP:
            name = get_workload_name(args.dataset, len(query_ids), args.n_neighborhoods, args.use_norm, args.objective, args.relabel_mode, True) # load from hetero dataset
            neighs_pyg_dict['train'] = LRP_Dataset(args.dataset, neighs_pyg_dict['train'], labels= [g.y for g in neighs_pyg_dict['train']], lrp_save_path=name, lrp_depth=1, subtensor_length=4, lrp_width=3)
        
        # valid
        dataset = test_dataset
        args.dataset_name = dataset
        args.dataset = dataset
        args.hetero = HETERO
        workload = get_workload(query_ids, args)
        neighs_pyg_dict['valid'] = workload.neighs_pyg
        print("begin training")
        if LRP:
            name = get_workload_name(args.dataset, len(query_ids), args.n_neighborhoods, args.use_norm, args.objective, args.relabel_mode, True)
            neighs_pyg_dict['valid'] = LRP_Dataset(args.dataset, neighs_pyg_dict['valid'], labels= [g.y for g in neighs_pyg_dict['valid']], lrp_save_path=name, lrp_depth=1, subtensor_length=4, lrp_width=3)

        train_canonical(model, query_ids, neighs_pyg_dict, args)

    if test:
        print("begin evaluating")
        # set params for canonical analyze
        for epoch in range(SAVE_EVERY_EPOCH, args.num_epoch+1, SAVE_EVERY_EPOCH):
            model_path = args.model_path+"_epo"+str(epoch)+".pt"
            dataset = test_dataset
            args.batch_size = 256
            args.dataset_name = dataset
            args.dataset = dataset
            args.hetero = HETERO
            test_workload = get_workload(query_ids, args, load_list= ['neighs_pyg', 'graphs_nx'])
            
            model_name = "/".join(model_path.split('/')[1:])
            model_name = model_name.split('.')[0]
            print(model_name)
            # norm_file = "/home/futy18/repos/Local/GNN/2021Summer/subgraph_counting/workload/syn_motif_n_query_29_n_neighs_57600_log_norm"
            model.load_state_dict(torch.load(model_path)['state_dict'])

            neighs_test = test_workload.neighs_pyg
            if GRAPHLET:
                neighs_index = [(gid,-1) for gid in range(len(neighs_test))] 
            else:
                neighs_index = test_workload.neighs_index
            
            if LRP:
                name = get_workload_name(args.dataset, len(query_ids), args.n_neighborhoods, args.use_norm, args.objective, args.relabel_mode, args.hetero)
                neighs_test = LRP_Dataset(args.dataset, neighs_test, labels= [g.y for g in neighs_test], lrp_save_path=name, lrp_depth=1, subtensor_length=4, lrp_width=3)
            eval_canonical(model, query_ids, neighs_test, neighs_index, model_name, args)

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    main(train= True, valid= True, test= True, test_dataset='ENZYMES')
    # main(train= False, valid= False, test= True, test_dataset='COX2')
    # main(train= False, valid= False, test= True, test_dataset='MUTAG')
    # main(train= False, valid= False, test= True, test_dataset='CiteSeer')
    # main(train= False, valid= False, test= True, test_dataset='Cora')

    # for test_dataset in ["ENZYMES", "MUTAG", "CiteSeer"]:
    # # for test_dataset in ['P2P']:
    #     main(train= False, test= True, test_dataset=test_dataset)
