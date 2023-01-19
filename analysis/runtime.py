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

from subgraph_counting.config import parse_gossip, parse_neighborhood, parse_optimizer
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (
    GossipCountingModel,
    NeighborhoodCountingModel,
)
from subgraph_counting.transforms import NetworkxToHetero, ToTconvHetero, ZeroNodeFeat
from subgraph_counting.workload import Workload


graph_transform = ToTconvHetero()


parser = argparse.ArgumentParser(description="Neighborhood embedding arguments")
parse_optimizer(parser)
args_opt = parser.parse_args()

# define neighborhood counting model
parse_neighborhood(parser)
args_neighborhood = parser.parse_args()
neighborhood_model = NeighborhoodCountingModel(
    input_dim=1, hidden_dim=64, args=args_neighborhood
)

neighborhood_model = neighborhood_model.to_hetero_old(
    tconv_target=args_neighborhood.use_tconv, tconv_query=args_neighborhood.use_tconv
)

data = load_data("DD")
data = [pyg.utils.to_networkx(graph, to_undirected=True) for graph in data]
for g in data:
    for n in g.nodes:
        g.nodes[n]["type"] = "count"
        g.nodes[n]["feat"] = torch.zeros(1)
    g.nodes[0]["type"] = "canonical"
data = [NetworkxToHetero(graph) for graph in data]
data = [graph_transform(graph) for graph in data]

neighborhood_train_dataloader = DataLoader(data, batch_size=32, shuffle=False)

for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for batch in neighborhood_train_dataloader:
        neighborhood_model.emb_model(batch)
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
