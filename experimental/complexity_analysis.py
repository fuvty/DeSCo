import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import datetime
import decimal
import math
import re
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from subgraph_counting.config import (parse_gossip, parse_neighborhood,
                                      parse_optimizer)
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (GossipCountingModel,
                                               NeighborhoodCountingModel)
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
from subgraph_counting.workload import Workload

def main(dataset_name):
    print("dataset: {}".format(dataset_name))

    dataset = load_data(dataset_name)
    num_classes = dataset.num_classes

    # get the groundtruth count of standard queries in each graph
    query_ids = gen_query_ids(query_size= [3,4,5])
    workload = Workload(dataset, 'data/'+dataset_name, hetero_graph=True)
    if workload.exist_groundtruth(query_ids=query_ids):
        workload.canonical_count_truth = workload.load_groundtruth(query_ids=query_ids)
    else:
        workload.canonical_count_truth = workload.compute_groundtruth(query_ids= query_ids, save_to_file= True)
    workload.generate_pipeline_datasets(depth_neigh=4, neighborhood_transform=None) # depth can be set to any number larger than 3, won't matter

    complexity_funcs = [lambda x: x**2, lambda x: 2**x, lambda x: x*math.factorial(x)]

    info_dict = {'method': [], 'complexity': [], 'func': [], 'dataset': []}

    for i,complexity_func in enumerate(complexity_funcs):
    # complexity_func = lambda x: 2**x

        # get statistics for graphlet dataset
        graph_node_num = [g.num_nodes for g in dataset]
        complexity = sum([complexity_func(n) for n in graph_node_num])
        print("complexity of graphlet dataset: {:.2e}".format(decimal.Decimal(complexity))) 
        # print using scientific notation
        info_dict['method'].append('graphlet')
        info_dict['complexity'].append(complexity)
        info_dict['func'].append(i)
        info_dict['dataset'].append(dataset_name)


        # get statistics for neighborhood dataset
        graph_node_num = [g.num_nodes for g in workload.neighborhood_dataset]    
        complexity = sum([complexity_func(n) for n in graph_node_num])
        print("complexity of neighborhood dataset: {:.2e}".format(decimal.Decimal(complexity)))
        info_dict['method'].append('neighborhood')
        info_dict['complexity'].append(complexity)
        info_dict['func'].append(i)
        info_dict['dataset'].append(dataset_name)

    return pd.DataFrame(info_dict)



if __name__ == "__main__":
    df = pd.DataFrame()

    dataset_names = ["MUTAG", "COX2", "ENZYMES"]
    for dataset_name in dataset_names:
        df_new = main(dataset_name)
        # concatenate the results
        df = pd.concat([df, df_new], ignore_index=True)

    df.to_csv("complexity.csv")