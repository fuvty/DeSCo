"""Train the regression subgraph counting model"""
import os
import sys

import matplotlib

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import csv
import math
import pickle
import random
import time
import warnings
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import torch_geometric as pyg
import torch_geometric.transforms as T
from common import data, models, utils
from numpy.core.fromnumeric import mean
from subgraph_matching.config import parse_encoder
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from subgraph_counting.baseline.GNNSubstructure.LRP_dataset import (
    LRP_Dataset, collate_lrp_dgl_light_index_form_wrapper)
from subgraph_counting.config import parse_count
from subgraph_counting.models import (BaseLineModel, CanonicalCountModel,
                                      MotifCountModel, MultiTaskModel)
from subgraph_counting.transforms import (NetworkxToHetero, ToTCONV, ToTconvHetero, get_truth,
                                          to_device)
from subgraph_counting.workload import Workload, graph_atlas_plus

LRP = False
HETERO = True
TCONV_T = True
TCONV_Q = True
BATCH_SIZE = 64

L_QUERYLEN = 6 # queries larger than this will be treated as L

def VisualizeDistributionScatter(x, y, title, file_path):
    '''
    get the distribution and scatter for y regard to x, x should be integers
    both x and y should be 1-D list-like object
    '''
    # x_mean & y_mean
    x_mean = np.unique(x)
    y_mean = []
    for cur_x in x_mean:
        y_mean.append(np.mean(y[x==cur_x]))
    y_mean = np.array(y_mean)

    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # y_unique and count
        x_unique = np.array([])
        y_unique = np.array([])
        count_unique = np.array([])
        x_range = np.unique(x)
        for x_cur in x_range:
            y_cur, index, count = np.unique(y[x==x_cur], return_index= True, return_counts= True)
            x_unique = np.append(x_unique, np.repeat(x_cur ,len(y_cur)) )
            y_unique = np.append(y_unique, y_cur)
            count_unique = np.append(count_unique, count)

        # the scatter plot:
        # y = np.log2(y+1)
        # y_unique = np.log2(y_unique+1)

        ax.scatter(x_unique, y_unique, s=10*np.log2(count_unique+1), alpha=0.8)

        # now determine nice limits by hand:
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax/binwidth) + 1) * binwidth

        binwidth = 1
        max_x = np.max(x)
        bins = np.arange(1, max_x + binwidth, binwidth)

        # filter = (y!=0)
        # x = x[filter]
        # y = y[filter]

        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, orientation='horizontal') # , bins=bins, orientation='horizontal')
    
    # start with a square Figure
    fig = plt.figure(figsize=(10, 10))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy)

    # plot average y in scatter_hist
    # y_mean = np.log2(y_mean+1) # draw in log scale
    ax.plot(x_mean, y_mean, color= 'orange', linewidth= 3)

    plt.suptitle(title, fontsize=20)
    ax.set_xlabel('Query', fontsize=15)
    ax.set_ylabel('Error/Var', fontsize=15)
    plt.savefig(file_path)

def VisualizeDistributionViolin(data, title, file_path, xs=None, ylim=None, figsize=(30,15)):
    '''
    data: a matrix-like object with shape y*x, make a violin plot for each column of dataset
    get the distribution of y
    '''
    plt.rcParams.update({'font.size': 30})

    def violin_hist(data, xs, ax, ax_histy, ylim):
        # no labels
        ax_histy.tick_params(axis="y", labelleft=False)

        ax.violinplot(data, positions=xs, vert=True, showmeans=True, showextrema=True, showmedians=True)

        # now determine nice limits by hand:
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax/binwidth) + 1) * binwidth
        y = []
        for d in data:
            y.extend(list(d))
        y = np.array(y)
        weights = np.ones_like(y)/float(y.size)
        ax_histy.hist(y, orientation='horizontal', weights=weights, bins=100, range=ylim) # , bins=bins, orientation='horizontal')
    
    # start with a square Figure
    fig = plt.figure(figsize=figsize)

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(1, 2,  width_ratios=(3, 1), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_ylim(ylim)
    ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)

    # use the previously defined function
    violin_hist(data, xs, ax, ax_histy, ylim)

    plt.suptitle(title, fontsize=40)
    ax.set_xlabel('Query Size', fontsize=30)
    ax.set_ylabel('Error/Var', fontsize=30)
    plt.savefig(file_path)


def eval_canonical(model, query_ids, neighs_pyg, neighs_index, model_name, args):
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

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
    
    ############### begin process ###############
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

    # dataset = load_data(args.dataset, args.n_neighborhoods)
    model.eval()
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    
    # prepare dataset and workload
    if LRP:
        # get query LRP
        queries_pyg = [g.to('cpu') for g in queries_pyg]
        query_LRP = LRP_Dataset('queries_'+str(len(queries_pyg)), graphs= queries_pyg, labels= [torch.zeros(1) for _ in range(len(queries_pyg))], lrp_save_path= '/home/nfs_data/futy/repos/prime/GNN_Mining/2021Summer/tmp_folder', lrp_depth=1, subtensor_length=4, lrp_width=3)

        query_loader = torch_data.DataLoader(query_LRP, batch_size=BATCH_SIZE, shuffle= False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(4))

        neighs_loader = torch_data.DataLoader(neighs_train, batch_size= BATCH_SIZE, shuffle=False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(4))
    else:
        query_loader = DataLoader(queries_pyg, batch_size= BATCH_SIZE)
        neighs_loader = DataLoader(neighs_train, batch_size= BATCH_SIZE)

    # inference
    print(model.emb_with_query)

    start_time = time.time()
    with torch.no_grad():
        # loss_query = [[] for _ in range(len(query_ids))]
        count_inference = [[] for _ in range(len(query_ids))]
        label_inference = [[] for _ in range(len(query_ids))]
        count_truth = [[] for _ in range(len(query_ids))]
        for batch_pyg in tqdm(neighs_loader):
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
            # end {inference}, commonly used for all

                # use relu to make sure positivity
                # if args.count_type == "motif":
                #     results = F.relu(results)
                # elif args.count_type == "multitask":  
                #     results['counting'] = F.relu(results['counting'])
                # else:
                #     raise NotImplementedError

                # expand to analyzed vars
                count_truth[i].append(truth)
                if args.count_type == "multitask":
                    count_inference[i].append(results['counting'])
                    try:
                        label_inference[i].append(torch.unsqueeze(torch.argmax(results['classification'], dim=-1), dim=-1))
                    except:
                        label_inference[i].append(torch.zeros_like(results['counting']))
                    # label_query.append(results[1], dim=-1)
                elif args.count_type == "motif":
                    count_inference[i].append(results)
                    label_inference[i].append(torch.zeros_like(results))

    end_time = time.time()
    print('Neighborhood Counting time: {:.4f}'.format(end_time - start_time))
    ######## end evaluation ########

    # inference count and label with size (#queries, #neighborhoods)
    # count_truth = torch.cat(count_truth, dim=0).T # .view(len(query_ids), -1)
    count_truth = torch.stack([torch.cat(count_truth[i], dim=0).view(-1) for i in range(len(query_ids))], dim=0) # shape: (#queries, #neighborhoods)
    count_inference = torch.stack([torch.cat(count_inference[i], dim=0).view(-1) for i in range(len(query_ids))], dim=0) # shape: (#queries, #neighborhoods)
    label_inference = torch.stack([torch.cat(label_inference[i], dim=0).view(-1) for i in range(len(query_ids))], dim=0) # # shape: (#queries, #neighborhoods)

    print("use norm", args.use_norm)
    assert(args.use_norm == False)
    if False: # use norm
        print("load norm value from "+norm_file)
        with open(norm_file, 'rb') as f:
            workload_norm = pickle.load(f)
            mean = workload_norm.norm_dict['mean_train'].view(-1,1,1,1).to(device)
            std = workload_norm.norm_dict['std_train'].view(-1,1,1,1).to(device)
        # count_truth = count_truth*std + mean 
        # the ground truth of testbench is log(count+1)
        count_inference = count_inference*std + mean
    
    print("use log", args.use_log)
    # use relu for inference results
    if args.use_log:
        label_truth = torch.floor(count_truth)
        count_truth = 2**count_truth - 1
        count_inference = 2**F.relu(count_inference) - 1
    else:
        label_truth = torch.floor(torch.log(count_truth+1))
        count_truth = count_truth
        count_inference = F.relu(count_inference)

    count_inference = torch.round(count_inference)

    analyze_neighborhood_counts(count_inference, count_truth, query_ids, neighs_index, model_name, args.dataset, label_inference, label_truth, device, False)

def analyze_neighborhood_counts(count_inference, count_truth, query_ids, neighs_index, model_name: str, dataset: str, label_inference= None, label_truth= None, device= 'cuda', compute_confusion_matrix= False):
    '''
    shape: (#queries, #neighborhoods)
    '''
    
    # assign results to each graph in dataset
    graph_id = torch.tensor([gid for gid,_ in neighs_index], dtype=torch.long).to(device)
    count_truth_graphlet = torch.zeros((len(query_ids), len(torch.unique(graph_id))), dtype=torch.float).to(device)
    count_inference_graphlet = torch.zeros((len(query_ids), len(torch.unique(graph_id))), dtype=torch.float).to(device)
    count_truth_graphlet.index_add_(dim=1, index=graph_id, source=count_truth)
    count_inference_graphlet.index_add_(dim=1, index=graph_id, source=count_inference)
    # save true count of dataset in graphlet manner
    pd.DataFrame(count_truth_graphlet.T.cpu().numpy()).to_csv(dataset+'_groundtruth.csv')

    raw_file = "results/raw/" + model_name + "_" + dataset + ".csv"
    if not os.path.exists(os.path.dirname(raw_file)):
        os.makedirs(os.path.dirname(raw_file))
    pd.DataFrame(count_inference_graphlet.T.cpu().numpy()).to_csv(raw_file)

    # compute error
    error_label = (label_truth!=label_inference)
    error_label = error_label.view(error_label.shape[0],-1)

    error_count = torch.abs(count_inference-count_truth)
    error_count = error_count.view(error_count.shape[0],-1)

    error_count_graphlet = torch.abs(count_inference_graphlet-count_truth_graphlet)
    error_count_graphlet = error_count_graphlet.view(error_count_graphlet.shape[0],-1)

    count_mse_query = torch.mean(error_count_graphlet**2, dim=1)
    count_var_query = torch.var(count_truth_graphlet, dim=1)
    count_mae_query = torch.mean(error_count_graphlet, dim=1)

    # visulize distribution of error
    # y = (count_inference_graphlet-count_truth_graphlet).view(len(query_ids), -1).cpu().numpy()
    # var = count_var_query.cpu().numpy().reshape(-1,1)
    # y = (y/var)
    # y_lim = (-0.1,0.1)
    # y = np.clip(y, y_lim[0], y_lim[1])
    # VisualizeDistributionScatter(x.reshape(-1), y.reshape(-1), args.dataset+'_error_distribution.png', 'tmp.jpg')
    # VisualizeDistributionViolin(y.T, args.dataset+'_error_distribution', 'plots/general/tmp1.jpg', xs= None, ylim=y_lim, figsize= (30,15))

    # group by number of nodes of queires
    query_lens = np.array([len(graph_atlas_plus(query_id)) for query_id in query_ids])
    query_lens.sort()
    query_lens_unique = np.unique(query_lens) # query_len: continues id
    # query_lens_unique_dict = {length:i for i,length in enumerate(query_lens_unique_dict)}
    # query length larger than L_QUERYLEN will be assigned to same group
    max_i = len(query_lens_unique[query_lens_unique<L_QUERYLEN])
    query_lens_unique_dict = {length:i if length<L_QUERYLEN else max_i for i,length in enumerate(query_lens_unique)}
    query_lens_unique = query_lens_unique[0:max_i+1]
    # list of tensors, each tensor is a group of query counts with the same number of nodes
    count_inference_graphlet_querylen = [torch.tensor([]).to(device) for _ in query_lens_unique]
    count_truth_graphlet_querylen = [torch.tensor([]).to(device) for _ in query_lens_unique]

    for qid, query_len in enumerate(query_lens):
        i = query_lens_unique_dict[query_len]
        count_inference_graphlet_querylen[i] = torch.cat((count_inference_graphlet_querylen[i], count_inference_graphlet[qid,:]), dim=0)
        count_truth_graphlet_querylen[i] = torch.cat((count_truth_graphlet_querylen[i], count_truth_graphlet[qid,:]), dim=0)

    # list of numpy arrays, each array is a group of query counts with the same number of nodes
    count_error_querylen = []

    # list of float numbers, each array is a group of query statistics with the same number of nodes
    count_mse_querylen = []
    count_var_querylen = []

    for i, _ in enumerate(query_lens_unique):
        count_mse_querylen.append(torch.mean((count_inference_graphlet_querylen[i]-count_truth_graphlet_querylen[i])**2).cpu().item())
        count_var_querylen.append(torch.var(count_truth_graphlet_querylen[i]).cpu().item())

        count_error_querylen.append((count_inference_graphlet_querylen[i]-count_truth_graphlet_querylen[i]).cpu().numpy())

    print("count_mse_querylen", count_mse_querylen)
    print("count_var_querylen", count_var_querylen)
    # visulize distribution of error by query length
    y = count_error_querylen
    var = count_var_querylen
    y = [y[i]/var[i] for i in range(len(var))]
    y_lim = (-0.1,0.1)
    y = [np.clip(y[i], y_lim[0], y_lim[1]) for i in range(len(y))]
    VisualizeDistributionViolin(y, dataset+'_error_distribution', 'plots/general/'+'_'.join((dataset,model_name.split('/')[-1],'.jpg')), xs=list(query_lens_unique), ylim=y_lim, figsize= (30,15))

    # for multitask model, analyze confusion matrix of Label and Count
    def label_count_confusion_matrix_analysis(truth, count, label) -> torch.Tensor:
        '''
        input: ground_truth count tensor, inference count tensor and predict label tensor
        output: confusion matrix
            [ inference Count catogory true, false].T * [ predict Label true, false], a.k.a
            [
                [[ CTLT, CTLF ],
                 [ CFLT, CFLF ]],
                ...
            ]
        '''
        truth = truth.view(truth.shape[0], -1)
        count = torch.floor(torch.log2(count+1)).view(count.shape[0],-1) # convert count into log2 catogory
        label = label.view(label.shape[0], -1)
        count_true = count==truth
        label_true = label==truth
        CTLT = torch.sum(count_true*label_true, dim= 1)
        CFLT = torch.sum((~count_true)*label_true, dim= 1)
        CTLF = torch.sum(count_true*(~label_true), dim= 1)
        CFLF = torch.sum((~count_true)*(~label_true), dim= 1)
        return torch.stack((torch.stack((CTLT,CTLF), dim=1),torch.stack((CFLT,CFLF), dim=1)), dim=1)

    if compute_confusion_matrix:
        label_count_confusion_matrix = label_count_confusion_matrix_analysis(label_truth, count_inference, label_inference)

    # sum count
    sum_truth = torch.sum(count_truth.view(count_truth.shape[0],-1), dim=-1)
    sum_inference = torch.sum(count_inference.view(count_inference.shape[0],-1), dim=-1)

    sum_truth_list = list(sum_truth.cpu().detach().numpy())
    sum_inference_list = list(sum_inference.cpu().detach().numpy())

    print("truth_graph", sum_truth_list)
    print("inference_graph", sum_inference_list)

    sum_of_error = np.sum(np.round(torch.abs(sum_inference-sum_truth).cpu().detach().numpy()))
    print("sum of error",sum_of_error)

    error_label_list = list((torch.sum(error_label, dim=-1)/error_label.shape[-1]).cpu().detach().numpy())
    error_count_list = list(torch.mean(error_count,dim=-1).cpu().detach().numpy())

    count_mse_var_querylen = [count_mse_querylen[i]/count_var_querylen[i] for i in range(len(count_var_querylen))]
    
    print("error_label", error_label_list)
    print("error_count", error_count_list)
    if compute_confusion_matrix:
        print("confusion_matrix", label_count_confusion_matrix)
        print("confusion_matrix_CFLT", label_count_confusion_matrix[:,1,0])

    csv_file = "results/"+ model_name + "_" + dataset + ".csv"

    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    with open(csv_file, 'w') as f:
        print("save results to ",csv_file)
        writer = csv.writer(f, dialect= 'excel')
        writer.writerow(["truth_graph"] + [str(n) for n in sum_truth_list])
        writer.writerow(['inference_graph'] + [str(n) for n in sum_inference_list])    
        writer.writerow(['error_count'] + [str(n) for n in error_count_list])   
        
        writer.writerow(['mae_graphlet'] + [str(n.item()) for n in count_mae_query])
        writer.writerow(['mse_graphlet'] + [str(n.item()) for n in count_mse_query]) 
        writer.writerow(['var_graphlet'] + [str(n.item()) for n in count_var_query])

        if compute_confusion_matrix:
            writer.writerow(['error_label'] + [str(n) for n in error_label_list])
            writer.writerow(["CTLT"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,0,0]])
            writer.writerow(["CTLF"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,0,1]])
            writer.writerow(["CFLT"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,1,0]])
            writer.writerow(["CFLF"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,1,1]])
        
        writer.writerow(['mse_graphlet_querylen']+[str(n) for n in count_mse_querylen])
        writer.writerow(['var_graphlet_querylen']+[str(n) for n in count_var_querylen])
        writer.writerow(['mse/var_graphlet_querylen']+[str(n) for n in count_mse_var_querylen])
        writer.writerow(['sum_of_error']+[str(sum_of_error)])


def analyze_graphlet_counts(count_inference_graphlet, count_truth_graphlet, query_ids, model_name: str, dataset: str, device= 'cuda', compute_confusion_matrix= False):
    '''
    shape: (#queries, #graphs)
    '''
    raw_file = "results/raw/" + model_name + "_" + dataset + ".csv"
    if not os.path.exists(os.path.dirname(raw_file)):
        os.makedirs(os.path.dirname(raw_file))
    pd.DataFrame(count_inference_graphlet.T.cpu().numpy()).to_csv(raw_file)

    error_count_graphlet = torch.abs(count_inference_graphlet-count_truth_graphlet)
    error_count_graphlet = error_count_graphlet.view(error_count_graphlet.shape[0],-1)

    count_mse_query = torch.mean(error_count_graphlet**2, dim=1)
    count_var_query = torch.var(count_truth_graphlet, dim=1)
    count_mae_query = torch.mean(error_count_graphlet, dim=1)

    # visulize distribution of error
    # y = (count_inference_graphlet-count_truth_graphlet).view(len(query_ids), -1).cpu().numpy()
    # var = count_var_query.cpu().numpy().reshape(-1,1)
    # y = (y/var)
    # y_lim = (-0.1,0.1)
    # y = np.clip(y, y_lim[0], y_lim[1])
    # VisualizeDistributionScatter(x.reshape(-1), y.reshape(-1), args.dataset+'_error_distribution.png', 'tmp.jpg')
    # VisualizeDistributionViolin(y.T, args.dataset+'_error_distribution', 'plots/general/tmp1.jpg', xs= None, ylim=y_lim, figsize= (30,15))

    # group by number of nodes of queires
    query_lens = np.array([len(graph_atlas_plus(query_id)) for query_id in query_ids])
    query_lens.sort()
    query_lens_unique = np.unique(query_lens) # query_len: continues id
    # query_lens_unique_dict = {length:i for i,length in enumerate(query_lens_unique_dict)}
    # query length larger than L_QUERYLEN will be assigned to same group
    max_i = len(query_lens_unique[query_lens_unique<L_QUERYLEN])
    query_lens_unique_dict = {length:i if length<L_QUERYLEN else max_i for i,length in enumerate(query_lens_unique)}
    query_lens_unique = query_lens_unique[0:max_i+1]
    # list of tensors, each tensor is a group of query counts with the same number of nodes
    count_inference_graphlet_querylen = [torch.tensor([]).to(device) for _ in query_lens_unique]
    count_truth_graphlet_querylen = [torch.tensor([]).to(device) for _ in query_lens_unique]

    for qid, query_len in enumerate(query_lens):
        i = query_lens_unique_dict[query_len]
        count_inference_graphlet_querylen[i] = torch.cat((count_inference_graphlet_querylen[i], count_inference_graphlet[qid,:]), dim=0)
        count_truth_graphlet_querylen[i] = torch.cat((count_truth_graphlet_querylen[i], count_truth_graphlet[qid,:]), dim=0)

    # list of numpy arrays, each array is a group of query counts with the same number of nodes
    count_error_querylen = []

    # list of float numbers, each array is a group of query statistics with the same number of nodes
    count_mse_querylen = []
    count_var_querylen = []

    for i, _ in enumerate(query_lens_unique):
        count_mse_querylen.append(torch.mean((count_inference_graphlet_querylen[i]-count_truth_graphlet_querylen[i])**2).cpu().item())
        count_var_querylen.append(torch.var(count_truth_graphlet_querylen[i]).cpu().item())

        count_error_querylen.append((count_inference_graphlet_querylen[i]-count_truth_graphlet_querylen[i]).cpu().numpy())

    print("count_mse_querylen", count_mse_querylen)
    print("count_var_querylen", count_var_querylen)
    # visulize distribution of error by query length
    y = count_error_querylen
    var = count_var_querylen
    y = [y[i]/var[i] for i in range(len(var))]
    y_lim = (-0.1,0.1)
    y = [np.clip(y[i], y_lim[0], y_lim[1]) for i in range(len(y))]
    VisualizeDistributionViolin(y, dataset+'_error_distribution', 'plots/general/'+'_'.join((dataset,model_name.split('/')[-1],'.jpg')), xs=list(query_lens_unique), ylim=y_lim, figsize= (30,15))

    # for multitask model, analyze confusion matrix of Label and Count
    def label_count_confusion_matrix_analysis(truth, count, label) -> torch.Tensor:
        '''
        input: ground_truth count tensor, inference count tensor and predict label tensor
        output: confusion matrix
            [ inference Count catogory true, false].T * [ predict Label true, false], a.k.a
            [
                [[ CTLT, CTLF ],
                 [ CFLT, CFLF ]],
                ...
            ]
        '''
        truth = truth.view(truth.shape[0], -1)
        count = torch.floor(torch.log2(count+1)).view(count.shape[0],-1) # convert count into log2 catogory
        label = label.view(label.shape[0], -1)
        count_true = count==truth
        label_true = label==truth
        CTLT = torch.sum(count_true*label_true, dim= 1)
        CFLT = torch.sum((~count_true)*label_true, dim= 1)
        CTLF = torch.sum(count_true*(~label_true), dim= 1)
        CFLF = torch.sum((~count_true)*(~label_true), dim= 1)
        return torch.stack((torch.stack((CTLT,CTLF), dim=1),torch.stack((CFLT,CFLF), dim=1)), dim=1)

    # if compute_confusion_matrix:
    #     label_count_confusion_matrix = label_count_confusion_matrix_analysis(label_truth, count_inference, label_inference)

    # sum count

    count_mse_var_querylen = [count_mse_querylen[i]/count_var_querylen[i] for i in range(len(count_var_querylen))]

    csv_file = "results/"+ model_name + "_" + dataset + ".csv"

    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    with open(csv_file, 'w') as f:
        print("save results to ",csv_file)
        writer = csv.writer(f, dialect= 'excel')

        writer.writerow(['mae_graphlet'] + [str(n.item()) for n in count_mae_query])
        writer.writerow(['mse_graphlet'] + [str(n.item()) for n in count_mse_query]) 
        writer.writerow(['var_graphlet'] + [str(n.item()) for n in count_var_query])

        # if compute_confusion_matrix:
        #     writer.writerow(['error_label'] + [str(n) for n in error_label_list])
        #     writer.writerow(["CTLT"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,0,0]])
        #     writer.writerow(["CTLF"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,0,1]])
        #     writer.writerow(["CFLT"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,1,0]])
        #     writer.writerow(["CFLF"] + [str(int(n.item())) for n in label_count_confusion_matrix[:,1,1]])
        
        writer.writerow(['mse_graphlet_querylen']+[str(n) for n in count_mse_querylen])
        writer.writerow(['var_graphlet_querylen']+[str(n) for n in count_var_querylen])
        writer.writerow(['mse/var_graphlet_querylen']+[str(n) for n in count_mse_var_querylen])

if __name__ == '__main__':
    device = 'cuda'

    queries = []
    atlas_graph = defaultdict(list)
    for i in range(4, 1253):
        g = graph_atlas_plus(i) # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_graph[len(g)].append(i)
    query_ids = atlas_graph[3] # + atlas_graph[4] + atlas_graph[5] # + atlas_graph[6] + atlas_graph[7]

    # query_ids += [83, 103, 286, 320, 8000, 8001, 9000, 9001, 10000, 10001, 11000, 11001, 12000, 12001, 13000, 13001, 14000, 14001]
    
    # query_ids += [81, 103, 276, 320, 8006, 8007, 9006, 9007, 10006, 10007, 11006, 11007, 12006, 12007, 13006, 13007]

    # set params for canonical analyze
    model_path = "ckpt/general/motif/sage_345_synXL_qs_triTQ_hetero_epo300.pt"
    evaluate_workload_file = "subgraph_counting/workload/general/CiteSeer_gossip_n_query_2_all_hetero"
    
    class Args:
        def __init__(self) -> None:
            # self.dataset = "ENZYMES"
            # self.dataset = "COX2"
            # self.dataset = "MUTAG"
            # self.dataset = "Cora"
            self.dataset = 'CiteSeer'
            self.conv_type = "SAGE" # only affect the final analysis, can be set wrongly
            self.dataset_name = self.dataset
            self.n_neighborhoods = 6400 # needed by syn
            self.batch_size = 64
            self.gpu = device
            self.count_type = "motif"
            self.use_log = True
            self.use_norm = False
            self.objective = "canonical"
            self.relabel_mode = None
    

    model_info = torch.load(model_path, map_location= 'cpu')
    model = BaseLineModel(1, 64, model_info['args'])

    if HETERO:
        if TCONV_T:
            model.emb_model.gnn_core = pyg.nn.to_hetero(model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union_triangle', 'count'), ('count', 'union_tride', 'count'), ('count', 'union_triangle', 'canonical'), ('count', 'union_tride', 'canonical'), ('canonical', 'union_triangle', 'count'), ('canonical', 'union_tride', 'count')] ) )
        else:
            model.emb_model.gnn_core = pyg.nn.to_hetero(model.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union', 'canonical'), ('canonical', 'union', 'count'), ('count', 'union', 'count')] ) )
        
        if TCONV_Q:
            model.emb_model_query.gnn_core = pyg.nn.to_hetero(model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union_triangle', 'union_node'), ('union_node', 'union_tride', 'union_node')] ) )
        else:
            model.emb_model_query.gnn_core = pyg.nn.to_hetero(model.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union', 'union_node')] ) )
    
    model.load_state_dict(model_info['state_dict'])
    model_name = model_path.split('/')[-1].split('.')[0]
    # norm_file = "/home/futy18/repos/Local/GNN/2021Summer/subgraph_counting/workload/syn_motif_n_query_29_n_neighs_57600_log_norm"

    model = model.to(device)

    args = Args()
    
    print("load ground truth from "+evaluate_workload_file)
    workload = Workload(evaluate_workload_file, sample_neigh= False, hetero_graph= HETERO)
    workload.load(evaluate_workload_file, load_list= ['neighs_pyg', 'graphs_nx'])
    print('begin evaluation')

    eval_canonical(model, query_ids, workload.neighborhood_dataset, workload.neighs_index, model_name, args)

    print("done")
