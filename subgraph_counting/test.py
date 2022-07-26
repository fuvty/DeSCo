import os
import random
import sys
from collections import defaultdict
from networkx.classes.function import degree
from networkx.generators import atlas

from numpy.core.fromnumeric import mean

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import multiprocessing as mp

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

if __name__ == '__main__':
    atlas_index = defaultdict(list)
    atlas_graph = defaultdict(list)
    count = 0
    '''
    for i in range(4, 1253):
        g = nx.graph_atlas(i) # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_index[len(g)].append(i)
            atlas_graph[len(g)].append(g)
            count += 1
    '''

    # d-regular graph
    for i in range(4, 1253):
        g = nx.graph_atlas(i) # range(0,1253)
        degree_set = set(g.degree(n) for n in g.nodes)
        if (len(degree_set)==1) and (degree_set!={0}) and (sum(1 for _ in nx.connected_components(g))==1):
            atlas_index[len(g)].append(i)
            atlas_graph[len(g)].append(g)
            count += 1
    
    g = nx.graph_atlas(11)
    nx.draw(g)
    plt.savefig('fty.png')
    print("done")

