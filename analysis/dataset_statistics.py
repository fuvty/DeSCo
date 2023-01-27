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
baseline_datasets = ["IMDB-BINARY", "MSRC-21", "ENZYMES", "COX2", "MUTAG"] + [
    "CiteSeer",
    "Cora",
    "FIRSTMM-DB",
]
# baseline_datasets = ['ENZYMES', 'CiteSeer']
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
# quick one

# %%
dataframe_dict = {"dataset": []}

dataframe_dict["dataset_name"] = []
dataframe_dict["depth"] = []
dataframe_dict["index"] = []

dataframe_dict["num_nodes"] = []
dataframe_dict["num_edges"] = []

for depth in depths:
    for index in indexes:
        for dataset in tqdm(datasets):
            dataset_name = dataset + index

            num_nodes = [G.num_nodes for G in pyg_neighs_dict[(dataset_name, depth)]]
            num_edges = [
                G.num_edges / 2 for G in pyg_neighs_dict[(dataset_name, depth)]
            ]

            dataframe_dict["dataset_name"].extend([dataset_name] * len(num_nodes))

            dataframe_dict["dataset"].extend([dataset] * len(num_nodes))
            dataframe_dict["depth"].extend([depth] * len(num_nodes))
            dataframe_dict["index"].extend([index] * len(num_nodes))

            dataframe_dict["num_nodes"].extend(num_nodes)
            dataframe_dict["num_edges"].extend(num_edges)

df = pd.DataFrame(dataframe_dict)

# %%
filename = "node-edge"
plt.figure(figsize=(16, 10))

df["dataset_class"] = df["dataset_name"] + "_" + df["depth"].astype(str)

sns.jointplot(
    x="num_nodes",
    y="num_edges",
    hue="dataset_class",
    data=df,
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
df["avg_degree"] = df["num_edges"] / df["num_nodes"]
for dataset_class_name in df["dataset_class"].unique():
    print(dataset_class_name)
    print(
        df[df["dataset_class"] == dataset_class_name].describe(
            percentiles=[0.5, 0.8, 0.9, 0.95, 0.99]
        )
    )

# %%
filename = "node-edge-comprehensive"
plt.figure(figsize=(16, 10))

df["dataset_class"] = df["dataset"] + "_" + df["depth"].astype(str)

sns.pairplot(
    # x="num_nodes", y="num_edges",
    hue="dataset_class",
    data=df,
    # alpha=0.3,
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
filename = "node-edge-reg"
plt.figure(figsize=(16, 10))

df["dataset_class"] = df["dataset"] + "_" + df["depth"].astype(str)

p = sns.lmplot(
    x="num_nodes",
    y="num_edges",
    hue="dataset",
    col="depth",
    row="index",
    data=df[df["dataset"] != synthetic_dataset],
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
filename = "node-degree-reg"
plt.figure(figsize=(16, 10))

df["dataset_class"] = df["dataset"] + "_" + df["depth"].astype(str)
df["avg_degree"] = df["num_edges"] / df["num_nodes"]

p = sns.jointplot(
    x="num_nodes",
    y="avg_degree",
    hue="dataset",
    data=df[df["dataset"] != synthetic_dataset],
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
filename = "degree-distri"
plt.figure(figsize=(16, 10))

df["dataset_class"] = df["dataset"] + "_" + df["depth"].astype(str)
df["avg_degree"] = df["num_edges"] / df["num_nodes"]

p = sns.jointplot(
    x="num_nodes",
    y="avg_degree",
    hue="dataset",
    data=df,
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
# regression analysis

# %%
for depth in depths:
    for index in indexes:
        for dataset in datasets:
            # use linear regression to fit the data
            x = df[
                (df["dataset"] == dataset)
                & (df["depth"] == depth)
                & (df["index"] == index)
            ]["num_nodes"].values.reshape(-1, 1)
            y = df[
                (df["dataset"] == dataset)
                & (df["depth"] == depth)
                & (df["index"] == index)
            ]["num_edges"].values.reshape(-1, 1)

            reg = LinearRegression(fit_intercept=False)
            reg.fit(x, y)

            # print the coefficients
            print(dataset, depth, index)
            print("Coefficients: ", reg.coef_)
            print("Intercept: ", reg.intercept_)
            print("R^2: ", reg.score(x, y))
            print()

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
    palette=sns.color_palette("hls", len(dataset_names)),
    markers="d",
    scale=1.3,
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

# %%
filename = "comprehensive"
plt.figure(figsize=(16, 10))

g = sns.pairplot(
    hue="dataset",
    palette=sns.color_palette("hls", len(dataset_names)),
    kind="scatter",
    plot_kws={"alpha": 0.5},
    data=df,
)


def hide_current_axis(*args, **kwds):
    plt.gca().cla()


g.map_upper(hide_current_axis)
g.map_upper(sns.kdeplot, levels=1, color=".2")

# save the figure in the output directory, if "tsne.png" already exists
# then append a number to the filename
i = 0
while True:
    full_name = os.path.join(output_dir, filename + "_" + str(i) + ".png")
    if not os.path.exists(full_name):
        break
    i += 1

print(full_name)
plt.savefig(full_name)
plt.savefig(full_name, bbox_inches="tight", format="pdf")

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
