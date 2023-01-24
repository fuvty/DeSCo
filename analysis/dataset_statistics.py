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
from torch_geometric.data import DataLoader
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
# baseline_datasets = ['IMDB-BINARY', 'ENZYMES', 'COX2', 'MUTAG', 'CiteSeer', 'P2P']
# baseline_datasets = ["ENZYMES", "ENZYMES_increaseByDegree", "ENZYMES_decreaseByDegree"]
baseline_datasets = ["COLORS-3"]
synthetic_dataset = "syn_4096"

dataset_names = baseline_datasets + [synthetic_dataset]

depth = 4

output_dir = "output/figures"

# %% [markdown]
# # Load data from file

# %%
nx_neighs_dict = dict()

for dataset_name in tqdm(dataset_names):
    dataset = NeighborhoodDataset(
        depth, "../data/" + dataset_name + "/NeighborhoodDataset"
    )
    nx_neighs_dict[dataset_name] = [
        pyg.utils.to_networkx(g.to_homogeneous(), to_undirected=True) for g in dataset
    ]

# %% [markdown]
# ## Statistic analysis

# %% [markdown]
# with pandas

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

for dataset_name in tqdm(nx_neighs_dict.keys()):
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
    dataframe_dict["clustering"].extend(clustering)
    dataframe_dict["shortest_path_length"].extend(path_length)
    dataframe_dict["diameter"].extend(diameter)
    dataframe_dict["density"].extend(density)
    dataframe_dict["num_nodes"].extend(num_nodes)
    dataframe_dict["num_edges"].extend(num_edges)
    dataframe_dict["avg_degree"].extend(avg_degree)

df = pd.DataFrame(dataframe_dict)

# %% [markdown]
# ### Pring average values

# %%
for dataset_name in dataset_names:
    print(dataset_name)
    print(df[df["dataset"] == dataset_name].describe())

# %%
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_list])

df["tsne-2d-one"] = tsne_results[:, 0]
df["tsne-2d-two"] = tsne_results[:, 1]

# %% [markdown]
# t-sne

# %%
filename = "tsne_realWorld"
plt.figure(figsize=(16, 10))
sns.jointplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue="dataset",
    palette=sns.color_palette("hls", len(dataset_names))[1:],
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


filename = "tsne_coverage"
plt.figure(figsize=(16, 10))
sns.jointplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue="dataset",
    palette=sns.color_palette("hls", len(dataset_names)),
    hue_order=[synthetic_dataset] + [n for n in baseline_datasets],
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

# %%
filename = "features"
plt.figure(figsize=(16, 10))

sns.despine(bottom=True, left=True)

melt_df = pd.melt(
    df,
    id_vars=["dataset"],
    value_vars=[
        "clustering",
        "shortest_path_length",
        "diameter",
        "density",
        "num_nodes",
        "num_edges",
    ],
)  # melt_df is a dataframe with 3 columns: dataset, variable, value; for example, the first row is COX2, clustering, 0.5

# normalize the value column
melt_df["value"] = melt_df.groupby(["variable"])["value"].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# show each observation with a scatterplot
sns.stripplot(
    x="value",
    y="variable",
    hue="dataset",
    data=melt_df,
    zorder=1,
    dodge=True,
    alpha=0.3,
    size=2,
    palette=sns.color_palette("hls", len(dataset_names)),
)

# show the conditional means
sns.pointplot(
    x="value",
    y="variable",
    hue="dataset",
    data=melt_df,
    join=False,
    palette="dark",
    markers="d",
    scale=0.75,
    errorbar=None,
    dodge=0.8 - 0.8 / len(dataset_names),
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

# %% [markdown]
# clustering & shortest path length

# %%
filename = "clustering-path_length"
plt.figure(figsize=(16, 10))

sns.jointplot(
    x="clustering",
    y="shortest_path_length",
    hue="dataset",
    palette=sns.color_palette("hls", len(dataset_names)),
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
plt.savefig(full_name)

# %% [markdown]
# diameter & density

# %%
filename = "diameter-density"
plt.figure(figsize=(16, 10))

sns.jointplot(
    x="diameter",
    y="density",
    hue="dataset",
    palette=sns.color_palette("hls", len(dataset_names)),
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

# %% [markdown]
# num of nodes & average degree

# %%
figurename = "node-degree"
plt.figure(figsize=(16, 10))

sns.jointplot(
    x="num_nodes",
    y="avg_degree",
    hue="dataset",
    palette=sns.color_palette("hls", len(dataset_names)),
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
