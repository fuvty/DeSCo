import os
import random
import sys
from collections import defaultdict

import torch_geometric as pyg
from networkx.generators import directed
from numpy.core.fromnumeric import mean

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import multiprocessing as mp
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import data, models, utils
from playground.lib.Anchor import SymmetricFactor
from subgraph_matching.config import parse_encoder
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from subgraph_counting.config import parse_count
from subgraph_counting.data import (OTFSynCanonicalDataSource, count_canonical,
                                    count_canonical_mp, sample_neigh_canonical,
                                    true_count_anchor)
from subgraph_counting.models import (CanonicalCountModel, MotifCountModel,
                                      MultiTaskModel)
from subgraph_counting.train import Workload

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Canonical Count')
    parse_count(parser)
    args = parser.parse_args()
    args.gpu = "cuda:4"
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    
    # settings
    num_query = 6
    args.count_type = "multitask"
    args.n_neighborhoods = 19200
    args.model_path = "ckpt/count_enzyme_4motif_log.pt"
    args.val_size = 6400
    args.use_log = True

    # nrow*ncol ~ num_query 
    nrows = 2
    ncols = 3

    # model
    model = torch.load(args.model_path)
    model = model.to(device)
    model.eval()
    
    # prepare dataset and workload
    name = "_" + args.count_type + "_" + "n_query_"+str(num_query) + "_" + "n_neighs_"+str(args.n_neighborhoods)+ "_" + args.model_path  
    if args.use_log:
        name += "_log"
    name = name.replace("/", "_")

    workload_file = "subgraph_counting/workload/" + name
    if os.path.exists(workload_file):
        print("load ground truth from "+workload_file)
        with open(workload_file, 'rb') as f:
            workload = pickle.load(f)
    else:
        raise NotImplementedError

    count_motif_train = workload.count_motif_train.to(device)
    count_motif_valid = workload.count_motif_valid.to(device)
    neighs_batch_train = workload.neighs_batch_train
    neighs_batch_valid = workload.neighs_batch_valid
    queries = workload.queries

    query_slice = slice(0,min([30,len(queries)]))
    queries = queries[query_slice]

    # valid using model
    print("validation: num queries ",len(queries))
    with torch.no_grad():
        count_motif_model = []
        class_motif_model = []
        for query_id in range(len(queries)):
            query = utils.batch_nx_graphs([queries[query_id] for _ in range(args.val_size)]).to(device)
            count_valid = count_motif_valid[query_id][0]

            emb_target = model.emb_model(neighs_batch_valid[0].to(device))
            if args.count_type == "motif":
                emb_query = model.emb_model_query(query)
                emb = torch.cat((emb_target, emb_query), dim=-1)
                count = model.count_model(emb)
                results = count
            elif args.count_type == "multitask":
                emb_query = model.emb_model_query(query)
                emb = torch.cat((emb_target, emb_query), dim=-1)
                count = model.count_model(emb)
                pred_label = model.classification(emb)
                results = (count, pred_label)
            elif args.count_type == "canonical":
                results = model.count_model(emb_target)
                results = results

            count_motif_model.append(torch.unsqueeze(count, dim=0))
            class_motif_model.append(torch.unsqueeze(pred_label.argmax(dim=-1).reshape(-1,1), dim=0))
            loss = model.criterion(results, count_valid)
            print("loss",loss)
    count_motif_model = torch.stack(count_motif_model, dim=0)  #shape = (query,1,batch_size,1)
    class_motif_model = torch.stack(class_motif_model, dim=0)#shape = (query,1,batch_size,1)

    batch_size = count_motif_valid.shape[2]

    # draw graph
    # take first
    count_motif_valid_np: np.ndarray = count_motif_valid[query_slice,:,:,:].cpu().numpy().reshape((len(queries), batch_size,1))
    count_motif_model_np: np.ndarray = count_motif_model.cpu().numpy().reshape((len(queries), batch_size,1))

    class_motif_valid_np: np.ndarray = np.floor(count_motif_valid_np) # for log usage, truth may not be an integer
    class_motif_valid_np[class_motif_valid_np>8] = 9
    class_motif_model_np: np.ndarray = class_motif_model.cpu().numpy().reshape((len(queries), batch_size,1))

    # draw sorted count
    seq = count_motif_valid_np.argsort(axis= 1)
    seq = seq[:,-1::-1,:]
    
    fig, axes = plt.subplots(nrows= nrows, ncols= ncols, figsize=(30,12))
    axes = axes.flatten()
    for i in range(len(queries)):
        ax = axes[i]
        if args.use_log:
            # power 2
            y_truth = np.power(2, count_motif_valid_np[i][seq[i,:,0]][:])-1 # shape (:,1)
            y_pred = np.power(2, count_motif_model_np[i][seq[i,:,0]][:])-1
        else:
            # regular count
            y_truth = count_motif_valid_np[i][seq[i,:,0]][:]
            y_pred = count_motif_model_np[i][seq[i,:,0]][:]

        ax.plot(y_truth, color='blue', label='ground_truth')
        ax.plot(y_pred, color='red', label='prediction')
        l1_loss_mean = np.mean(np.abs(y_truth-y_pred))
        ax.set_title("l1_loss_mean=" + str(l1_loss_mean))

    ax.legend()
    plt.savefig("results/subgraph_counting/sort_count.png")


    # draw sorted class
    fig, axes = plt.subplots(nrows= nrows, ncols= ncols, figsize=(30,12))
    axes = axes.flatten()
    for i in range(len(queries)):
        ax = axes[i]
        if False:
            # power 2
            y_truth = np.power(2, class_motif_valid_np[i][seq[i,:,0]][:])-1 # shape (:,1)
            y_pred = np.power(2, class_motif_model_np[i][seq[i,:,0]][:])-1
        else:
            # regular count
            y_truth = class_motif_valid_np[i][seq[i,:,0]][:]
            y_pred = class_motif_model_np[i][seq[i,:,0]][:]

        ax.plot(y_truth, color='blue', label='ground_truth')
        ax.plot(y_pred, color='red', label='prediction')
        l1_loss_mean = np.mean(np.abs(y_truth-y_pred))
        ax.set_title("l1_loss_mean=" + str(l1_loss_mean))

    ax.legend()
    plt.savefig("results/subgraph_counting/sort_class.png")

    # draw query pattern
    fig, axes = plt.subplots(nrows= nrows, ncols= ncols, figsize=(30,12))
    axes = axes.flatten()
    for i in range(len(queries)):
        nx.draw(queries[i], ax=axes[i])
        axes[i].set_title("symmetry_factor= "+ str(SymmetricFactor(queries[i])))
    plt.savefig("results/subgraph_counting/queries.png")

    # print count
    print("done")
    


