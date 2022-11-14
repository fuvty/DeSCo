import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import datetime
import re
from typing import List, Optional, Tuple, Union
import re

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
from subgraph_counting.workload import Workload, graph_atlas_plus


def to_gml(dataset, type_g, path):
    names = []

    for i,g in enumerate(dataset):
        for n in g.nodes:
            g.nodes[n]['label'] = 0
        for e in g.edges:
            g.edges[e]['label'] = 0
            g.edges[e]['key'] = 0
        name = '{}_N{}_E{}_NL1_EL1_{}.gml'.format(type_g,len(g.nodes), len(g.edges), i)
        names.append(name.split('.')[0])

        g.graph['directed'] = 0

        nx.write_gml(g, path + name)

        # replace all the labels in the file
        with open(path + name, 'r') as f:
            lines = f.readlines()
        with open(path + name, 'w') as f:
            for line in lines:
                line = re.sub('label "\d"', 'label 0', line)
                f.write(line)
                    
    return names


if __name__ == "__main__":
    dataset_name = "MUTAG"
    dataset = load_data(dataset_name)
    path = 'experimental/gml/{}/'.format(dataset_name)

    if not os.path.exists(path):
        os.makedirs(path+'graphs/')
        os.makedirs(path+'patterns/')
        os.makedirs(path+'metadata/')

    # convert graph to gml format
    targets = [pyg.utils.to_networkx(graph, to_undirected=True) for graph in dataset]
    graph_names = to_gml(targets, 'G', path+'graphs/')

    # convert query to gml format
    query_ids = gen_query_ids(query_size= [3,4,5])
    queries = [graph_atlas_plus(n) for n in query_ids]
    query_names = to_gml(queries, 'P', path+'patterns/')

    # get metadata
    for name_query, query in zip(query_names, queries):
        filename = path+'metadata/'+name_query+'.csv'
        with open(filename, 'w') as f:
            f.write('g_id,counts,subisomorphisms\n')

        # compute query number and matches
        query_num = 0
        matches = []
        for name_target, target in zip(graph_names, targets):
            GraphMatcher = nx.algorithms.isomorphism.GraphMatcher(target, query)
            SBM_iter = GraphMatcher.subgraph_isomorphisms_iter()
            for vmap in SBM_iter:
                matches.append(list(vmap.keys()))

            # write metadata with format
            # g_id,counts,subisomorphisms
            # G_N11_E22_NL3_EL3_16,12,"[[0, 1, 2], [0, 5, 4], [1, 0, 5], [1, 2, 3], [2, 1, 0], [2, 3, 4], [3, 2, 1], [3, 4, 5], [4, 3, 2], [4, 5, 0], [5, 0, 1], [5, 4, 3]]"
            with open(filename, 'a') as f:
                f.write(','.join([name_target, str(len(matches)), "\"{}\"".format(str(matches))])+'\n')
