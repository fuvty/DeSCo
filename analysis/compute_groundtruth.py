from operator import truth
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import datetime
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import pickle

from subgraph_counting.config import (parse_gossip, parse_neighborhood,
                                      parse_optimizer)
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (GossipCountingModel,
                                               NeighborhoodCountingModel)
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
from subgraph_counting.workload import Workload


def compute_groundtruth(train_dataset_name, output_path, force_nx_queries: List[nx.Graph] = None):
    '''
    train the model and test accorrding to the config
    '''

    # define queries
    if force_nx_queries is None:
        query_ids = gen_query_ids(query_size= [6])
        force_nx_queries = [nx.graph_atlas(i) for i in query_ids]
    else:
        query_ids = None

    print('use queries with atlas ids:', query_ids)

    # define training workload
    target_dataset = load_data(train_dataset_name)
    workload = Workload(target_dataset, 'data/'+train_dataset_name, hetero_graph=True)
    if workload.exist_groundtruth(query_ids=query_ids):
        workload.canonical_count_truth = workload.load_groundtruth(query_ids=query_ids)
    else:
        workload.canonical_count_truth = workload.compute_groundtruth(query_ids= query_ids, queries= force_nx_queries, save_to_file= True)
    workload.generate_pipeline_datasets(depth_neigh=1) # generate pipeline dataset, including neighborhood dataset and gossip dataset

    ground_truth_canonical = workload.canonical_count_truth[workload.neighborhood_dataset.nx_neighs_indicator,:]
    ground_truth_graphlet = workload.neighborhood_dataset.aggregate_neighborhood_count(ground_truth_canonical)

    truth_sum = torch.sum(ground_truth_graphlet, dim=0, dtype=int).detach().cpu().numpy().reshape(-1,1)
    query_size = np.array([len(g) for g in force_nx_queries], dtype=int).reshape(-1,1)
    output = np.concatenate((truth_sum, query_size), axis=1, dtype=int)
    # output = truth_sum

    print('ground truth sum:', output)

    pd.DataFrame(output).to_csv(output_path) # save the inferenced results to csv file

    

if __name__ == "__main__":
    query_path = '/home/futy18/nfs/repos/prime/GNN_Mining/freq_mining/results/out-patterns-mfinder.p'
    output_path = '/home/futy18/nfs/repos/prime/GNN_Mining/freq_mining/results/ground_truth_mfinder.csv'
    # output_path = '/home/futy18/nfs/repos/prime/NIPS22/DeSCo/data/ENZYMES/CanonicalCountTruth/ground_truth_size_6_DeSCo.csv'

    with open(query_path, 'rb') as f:
        queries = pickle.load(f)

    compute_groundtruth('ENZYMES', output_path, force_nx_queries=queries)

    # compute_groundtruth('ENZYMES', output_path, force_nx_queries=None)
