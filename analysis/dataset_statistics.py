# %% [markdown]
# # Dataset Statistics Analysis
#

# %% [markdown]
# imports

# %%
import math
import os
import random
import sys
from collections import defaultdict
from ctypes.wintypes import INT
from typing import List
import csv

import torch_geometric as pyg
from networkx.generators import directed
from numpy.core.fromnumeric import mean

# parentdir = os.path.dirname(os.path.realpath('./'))
# sys.path.append(parentdir)

import argparse
import multiprocessing as mp
import pickle
import warnings
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from tqdm.contrib import tzip

from sklearn.manifold import TSNE


from subgraph_counting.data import (
    count_canonical,
    count_canonical_mp,
    count_graphlet,
    get_neigh_canonical,
    load_data,
    sample_graphlet,
    sample_neigh_canonical,
    true_count_anchor,
)
from subgraph_counting.workload import NeighborhoodDataset, GossipDataset, Workload

# %% [markdown]
# # Settings

# %%
baseline_datasets = ["IMDB-BINARY", "MSRC-21", "ENZYMES", "COX2", "MUTAG"] + [
    "CiteSeer",
    "Cora",
    "FIRSTMM-DB",
]
# baseline_datasets = ['ENZYMES', 'CiteSeer'] 'IMDB-BINARY', 'IMDB-BINARY_decreaseByDegree'
# baseline_datasets = ['P2P']
# baseline_datasets = ["MSRC-21", "FIRSTMM-DB"]
# baseline_datasets = ["CiteSeer", "CiteSeer_increaseByDegree", "CiteSeer_decreaseByDegree"]
# synthetic_dataset = 'Syn_128'
synthetic_dataset = "Syn_1827"

depth = 4
depths = [4]

index = ""
# indexes = ["", "_decreaseByDegree"]
indexes = [""]
# indexes = ["_decreaseByDegree"]

output_dir = "output/figures"


datasets = baseline_datasets + [synthetic_dataset]
# datasets = baseline_datasets

####################################################
dataset_names = []
for dataset in datasets:
    for index in indexes:
        dataset_names.append(dataset + index)

# %% [markdown]
# # Load data from file

# %%
pyg_neighs_dict = dict()

for depth in depths:
    for dataset_name in tqdm(dataset_names):
        dataset = NeighborhoodDataset(
            depth, "../data/" + dataset_name + "/NeighborhoodDataset"
        )
        pyg_neighs_dict[(dataset_name, depth)] = [
            g.to_homogeneous(add_node_type=False, add_edge_type=False) for g in dataset
        ]

# %% [markdown]
# ## Statistic analysis

# %% [markdown]
# # Graph Data analysis

# %%
# pyg_graphs_dict = dict()

# for depth in depths:
#     for dataset_name in tqdm(dataset_names):
#         dataset = load_data(dataset_name, root_folder='../data')
#         pyg_graphs_dict[dataset_name] = [g for g in dataset]

# %%
# nx_graphs_dict = dict()
# for dataset_name in tqdm(dataset_names):
#     nx_graphs_dict[dataset_name] = [pyg.utils.to_networkx(g, to_undirected=True) for g in pyg_graphs_dict[dataset_name]]

# %%
feat_list = [
    "nodes",
    "edges",
    "degree",
    "clustering",
    "shortest path",
    "diameter",
    "density",
]

# dataframe_dict = {'dataset':[]}
# for feat in feat_list:
#     dataframe_dict[feat] = []

# for dataset_name in tqdm(dataset_names):
#     graphs = [G.subgraph(max(nx.connected_components(G), key=len)) for G in nx_graphs_dict[dataset_name]]

#     num_nodes = [G.number_of_nodes() for G in graphs]
#     num_edges = [G.number_of_edges() for G in graphs]
#     clustering = [nx.average_clustering(G) for G in graphs]
#     path_length = [nx.average_shortest_path_length(G) for G in graphs]
#     diameter = [nx.diameter(G) for G in graphs]
#     density = [nx.density(G) for G in graphs]
#     avg_degree = [np.mean([d for n, d in G.degree()]) for G in graphs]

#     dataframe_dict['dataset'].extend([dataset_name]*len(graphs))

#     dataframe_dict['nodes'].extend(num_nodes)
#     dataframe_dict['degree'].extend(avg_degree)
#     dataframe_dict['edges'].extend(num_edges)

#     dataframe_dict['clustering'].extend(clustering)
#     dataframe_dict['shortest path'].extend(path_length)
#     dataframe_dict['diameter'].extend(diameter)
#     dataframe_dict['density'].extend(density)

# df = pd.DataFrame(dataframe_dict)
# # %%
# # save df to file, if the file already exists
# # then append a number to the filename
# i = 0
# filename =  "graph-features"
# while True:
#     full_name = os.path.join(output_dir, filename+"_"+str(i)+".csv")
#     if not os.path.exists(full_name):
#         break
#     i += 1
# df.to_csv(os.path.join(full_name))

# # %%
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(df[feat_list])

# df['tsne-x'] = tsne_results[:,0]
# df['tsne-y'] = tsne_results[:,1]

# # %%
# filename = "tsne"
# f = plt.figure(figsize=(16,10))
# ax = sns.jointplot(
#     x="tsne-x", y="tsne-y",
#     hue="dataset",
#     data=df[df['dataset']!=synthetic_dataset],
#     # data=df,
#     legend="full",
#     alpha=0.3,
# )
# # save the figure in the output directory, if "tsne.png" already exists
# # then append a number to the filename
# i = 0
# while True:
#     full_name = os.path.join(output_dir, filename+"_"+str(i)+".png")
#     if not os.path.exists(full_name):
#         break
#     i += 1
# plt.savefig(full_name, bbox_inches='tight')
# plt.savefig(full_name.replace(".png", ".pdf"), bbox_inches='tight', format='pdf')
# plt.close(f)

# # %%
# for dataset_name in dataset_names:
#     print(dataset_name)
#     print(df[df['dataset'] == dataset_name].describe())

# %% [markdown]
# ## Comprehensive analysis

# %%
nx_neighs_dict = dict()
for dataset_name in tqdm(dataset_names):
    nx_neighs_dict[dataset_name] = [
        pyg.utils.to_networkx(g, to_undirected=True)
        for g in pyg_neighs_dict[(dataset_name, depth)]
    ]

# %%
feat_list = [
    "clustering",
    "shortest_path_length",
    "diameter",
    "density",
    "num_nodes",
    "num_edges",
    "avg_degree",
]

dataframe_dict = {"dataset": []}
for feat in feat_list:
    dataframe_dict[feat] = []

for dataset_name in tqdm(dataset_names):
    graphs = [
        G.subgraph(max(nx.connected_components(G), key=len))
        for G in nx_neighs_dict[dataset_name]
    ]

    num_nodes = [G.number_of_nodes() for G in graphs]
    num_edges = [G.number_of_edges() for G in graphs]
    clustering = [nx.average_clustering(G) for G in graphs]
    path_length = [nx.average_shortest_path_length(G) for G in graphs]
    diameter = [nx.diameter(G) for G in graphs]
    density = [nx.density(G) for G in graphs]
    avg_degree = [np.mean([d for n, d in G.degree()]) for G in graphs]

    dataframe_dict["dataset"].extend([dataset_name] * len(graphs))

    dataframe_dict["num_nodes"].extend(num_nodes)
    dataframe_dict["avg_degree"].extend(avg_degree)
    dataframe_dict["num_edges"].extend(num_edges)

    dataframe_dict["clustering"].extend(clustering)
    dataframe_dict["shortest_path_length"].extend(path_length)
    dataframe_dict["diameter"].extend(diameter)
    dataframe_dict["density"].extend(density)

df = pd.DataFrame(dataframe_dict)

# %%
# save df to file, if the file already exists
# then append a number to the filename
i = 0
filename = "neighborhood-features"
while True:
    full_name = os.path.join(output_dir, filename + "_" + str(i) + ".csv")
    if not os.path.exists(full_name):
        break
    i += 1
df.to_csv(os.path.join(full_name))

# %% [markdown]
# ### Pring average values

# %%
for dataset_name in dataset_names:
    print(dataset_name)
    print(df[df["dataset"] == dataset_name].describe())

# %%
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_list])

df["t-SNE x"] = tsne_results[:, 0]
df["t-SNE y"] = tsne_results[:, 1]

# %% [markdown]
# t-sne

# %%
filename = "tsne_realWorld"
f = plt.figure(figsize=(16, 10))
sns.jointplot(
    x="t-SNE x",
    y="t-SNE y",
    hue="dataset",
    # palette=sns.color_palette("hls", len(dataset_names))[:-1],
    data=df[df["dataset"] != synthetic_dataset],
    legend="full",
    alpha=0.3,
)
# save the figure in the output directory, if "tsne.png" already exists
# then append a number to the filename
i = 0
while True:
    full_name = os.path.join(output_dir, filename + "_" + str(i) + ".png")
    if not os.path.exists(full_name):
        break
    i += 1
plt.savefig(full_name, bbox_inches="tight")
plt.close(f)

filename = "tsne_coverage"
f = plt.figure(figsize=(16, 10))
sns.jointplot(
    x="t-SNE x",
    y="t-SNE y",
    hue="dataset",
    # palette=sns.color_palette("hls", len(dataset_names)),
    hue_order=[n for n in baseline_datasets] + [synthetic_dataset],
    data=df,
    legend="full",
    alpha=0.3,
)
# save the figure in the output directory, if "tsne.png" already exists
# then append a number to the filename
i = 0
while True:
    full_name = os.path.join(output_dir, filename + "_" + str(i) + ".png")
    if not os.path.exists(full_name):
        break
    i += 1
plt.savefig(full_name, bbox_inches="tight")
plt.close(f)
