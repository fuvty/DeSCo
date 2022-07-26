# %% [markdown]
# # Standard Subgraph Counting Analysis
# Analyze the subgraph counting related data
# Standard Count Analysis by iterating through all graphs and neighbors

# %%
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)


import argparse
import csv
import math
import multiprocessing as mp
import pickle
import warnings
from collections import defaultdict

import deepsnap as ds
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric as pyg
from common import data, models, utils
from matplotlib import cm
from playground.lib.Anchor import GenVMap, GetAnchoredGraph, SymmetricFactor
from sklearn.decomposition import PCA
from subgraph_matching.config import parse_encoder
from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from subgraph_counting.config import parse_count
from subgraph_counting.data import (count_canonical, count_canonical_mp,
                                    get_neigh_canonical, load_data,
                                    sample_neigh_canonical, true_count_anchor)
from subgraph_counting.models import CanonicalCountModel
from subgraph_counting.train import Workload

# %% [markdown]
# ## Config

# %% [markdown]
# Config what standard test to use

# %%
model_path = "/home/futy18/repos/Local/GNN/2021Summer/ckpt/baseline/sage_345_opt_graphlet_epo900.pt"
model = torch.load(model_path)

model_name = model_path.split('/')[-1].split('.')[0]

workload_name = "enzymes_b100_345_graphlet_all"

# norm_file = "/home/futy18/repos/Local/GNN/2021Summer/subgraph_counting/workload/syn_motif_n_query_29_n_neighs_57600_log_norm"
# %%
model.emb_model_query.conv_type

# %% [markdown]
# define args

# %%
class Args:
    def __init__(self) -> None:
        self.dataset = "ENZYMES"
        # self.dataset = "COX2"
        # self.dataset = "REDDIT-BINARY"
        self.conv_type = "SAGE" # only affect the final analysis, can be set wrongly
        self.dataset_name = self.dataset
        self.n_neighborhoods = 19200 # needed by syn
        self.batch_size = 100
        self.gpu = "cuda:0"
        self.count_type = "motif"
        self.use_log = True
        self.use_norm = False
        self.objective = "gossip"

args = Args()

# %% [markdown]
# define query

# %%
queries = []
atlas_graph = defaultdict(list)
for i in range(4, 1253):
    g = nx.graph_atlas(i) # range(0,1253)
    if sum(1 for _ in nx.connected_components(g)) == 1:
        atlas_graph[len(g)].append(g)
queries = atlas_graph[3] + atlas_graph[4] + atlas_graph[5] # + atlas_graph[6] # + atlas_graph[7]

# %%
# queries = four_node_queries

# counts = []
# for query in tqdm(queries):
#     # count
#     # count = true_count_anchor(query, dataset, num_worker=10)
#     # count = true_count_graph(query, dataset, num_worker=10)
#     count = sum(count_canonical_mp(query, dataset, num_worker=10, from_pyg= True))
#     counts.append(count)
# print(counts)

# %% [markdown]
# ## RUN

# %% [markdown]
# load dataset

# %%
len_neighbor =  max(nx.diameter(query) for query in queries)

# dataset = load_data(args.dataset, args.n_neighborhoods)
model.eval()
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# load/gen workload ground truth

# %%
workload_file = "workload/" + workload_name
if os.path.exists(workload_file):
    print("load ground truth from "+workload_file)
    with open(workload_file, 'rb') as f:
        workload = pickle.load(f)
else:
    workload = Workload(workload_name)
    print("generate and save ground truth to "+workload_file)
    if args.objective == "gossip":
        workload.gen_workload_general(query_ids, args)
    else:
        workload.gen_workload_all(queries, args)
    with open(workload_file, 'wb') as f:
        pickle.dump(workload, f)
# all data needed is prepared
# %%
count_motif_all = workload.canonical_count_truth.to(device)
neighs_batch_all = [b.to(device) for b in workload.neighborhood_dataset]
num_query = len(workload.queries)

# %% [markdown]
# inference

# %%
print(model.emb_model_query.conv_type, model.emb_with_query)

# %%
loss_all = []
count_all = []
label_all = []
with torch.no_grad():
    for query_id in tqdm(range(len(queries))):
        loss_query = []
        count_query = []
        label_query = []
        for b in range(len(neighs_batch_all)):
            batch = neighs_batch_all[b]
            query = utils.batch_nx_graphs([queries[query_id] for _ in range(args.batch_size)])
            truth = count_motif_all[query_id][b]

            if not hasattr(model, 'emb_with_query'):
                raise NotImplementedError
            if model.emb_with_query:
                emb_query = model.emb_model_query(query, query_emb= torch.zeros(model.hidden_dim).to(args.gpu))
                emb_target = model.emb_model(batch, query_emb= emb_query[0,:])
            else:
                emb_query = model.emb_model_query(query)
                emb_target = model.emb_model(batch)
            
            if model.emb_with_query:
                emb_query = model.emb_model_query(query, query_emb= torch.zeros(model.hidden_dim).to(args.gpu))
                emb_target = model.emb_model(batch, query_emb= emb_query[0,:])
            else:
                emb_query = model.emb_model_query(query)
                emb_target = model.emb_model(batch)

            if args.count_type == "motif":
                emb = (emb_target, emb_query)
                results = model.count(emb)
            elif args.count_type == "multitask":  
                emb = torch.cat((emb_target, emb_query), dim=-1)
                count = model.count_model(emb)
                pred_label = model.classification(emb)
                results = [count, pred_label]
            elif args.count_type == "canonical":
                results = model.count_model(emb_target)
            else:
                raise NotImplementedError
            loss = model.criterion(results, truth)
            
            loss_query.append(loss)
            if args.count_type == "multitask":
                count_query.append(results[0])
                label_query.append(torch.unsqueeze(torch.argmax(results[1], dim=-1), dim=-1))
                # label_query.append(results[1], dim=-1)
            elif args.count_type == "motif":
                count_query.append(results)
                label_query.append(torch.zeros_like(results))
            
        loss_all.append(sum(loss_query)/len(loss_query))
        count_all.append(torch.stack(count_query, dim=0))
        label_all.append(torch.stack(label_query, dim=0))

# %%
count_all = torch.stack(count_all, dim=0)
label_all = torch.stack(label_all, dim=0)

# %% [markdown]
# ## Analysis

# %% [markdown]
# standard score

# %%
count_truth = count_motif_all
count_inference = count_all

print("use norm", args.use_norm)
if args.use_norm:
    print("load norm value from "+norm_file)
    with open(norm_file, 'rb') as f:
        workload_norm = pickle.load(f)
        mean = workload_norm.norm_dict['mean_train'].view(-1,1,1,1).to(device)
        std = workload_norm.norm_dict['std_train'].view(-1,1,1,1).to(device)
    # count_truth = count_truth*std + mean 
    # the ground truth of testbench is log(count+1)
    count_inference = count_inference*std + mean
print("use log", args.use_log)
if args.use_log:
    label_truth = torch.floor(count_truth)
    count_truth = 2**count_truth - 1
    count_inference = F.relu(2**count_inference - 1)
else:
    raise NotImplementedError

count_inference = torch.round(count_inference)

error_label = (label_truth!=label_all)
error_label = error_label.view(error_label.shape[0],-1)

error_count = torch.abs(count_inference-count_truth)
error_count = error_count.view(error_count.shape[0],-1)

# %% [markdown]
# sum count

# %%
sum_truth = torch.sum(count_truth.view(count_truth.shape[0],-1), dim=-1)
sum_inference = torch.sum(count_inference.view(count_inference.shape[0],-1), dim=-1)

sum_truth_list = list(sum_truth.cpu().detach().numpy())
sum_inference_list = list(sum_inference.cpu().detach().numpy())

print("truth_graph", sum_truth_list)
print("inference_graph", sum_inference_list)

# %%
print("sum of error",np.round(np.sum(torch.abs(sum_inference-sum_truth).cpu().detach().numpy())))

# %%
error_label_list = list((torch.sum(error_label, dim=-1)/error_label.shape[-1]).cpu().detach().numpy())
error_count_list = list(torch.mean(error_count,dim=-1).cpu().detach().numpy())

print("error_label", error_label_list)
print("error_count", error_count_list)

# %% [markdown]
# write results as csv

# %%
with open("../results/baseline/"+ model_name + ".csv", 'w') as f:
    writer = csv.writer(f, dialect= 'excel')
    writer.writerow(["truth_graph"] + [str(n) for n in sum_truth_list])
    writer.writerow(['inference_graph'] + [str(n) for n in sum_inference_list])    
    writer.writerow(['error_count'] + [str(n) for n in error_count_list])    

# %% [markdown]
# draw the graphs

# %%
nrows = 6
ncols = math.ceil(num_query/nrows)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,10))
ax = axes.flatten()

queries = workload.queries
for i,graph in enumerate(queries):
    node_color = [ 'red' if anchor==1 else 'blue' for anchor in nx.get_node_attributes(graph, 'anchor').values() ]
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, ax=ax[i], with_labels=False, pos=pos, node_color=node_color)
    ax[i].title.set_text(str(i))
plt.suptitle(str(args.dataset_name))
plt.show()

# %%
nrows = 6
ncols = math.ceil(num_query/nrows)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,10))
ax = axes.flatten()

queries = workload.queries
for i,graph in enumerate(queries):
    node_color = [ 'red' if anchor==1 else 'blue' for anchor in nx.get_node_attributes(graph, 'anchor').values() ]
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, ax=ax[i], with_labels=False, pos=pos, node_color=node_color)
    ax[i].title.set_text(str(int(math.factorial(len(graph))/SymmetricFactor(graph))))
plt.suptitle(str(args.dataset))
plt.show()

# %% [markdown]
# convert error to numpy

# %%
error_label_np = error_label.cpu().numpy()
error_count_np = error_count.cpu().numpy()

truth_count_np = count_truth.view(count_truth.shape[0],-1).cpu().numpy()
model_count_np = count_inference.view(count_inference.shape[0],-1).cpu().numpy()

truth_label_np = label_truth.view(label_truth.shape[0],-1).cpu().numpy()
model_label_np = label_all.view(label_all.shape[0],-1).cpu().numpy()

# %% [markdown]
# draw error of each query

# %%
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,10))
axes = axes.flatten()
for i in range(len(queries)):
# for i in [0,1,2,3,4,5]:
    ax = axes[i]
    ax.plot(truth_label_np[i][0:100], color='blue', label='ground_truth')
    ax.plot(model_label_np[i][0:100], color='red', label='prediction')

ax.legend()
axes[1].set_title("pred label")

# %%
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,10))
axes = axes.flatten()
for i in range(len(queries)):
    ax = axes[i]
    ax.plot(truth_count_np[i][0:100], color='blue', label='ground_truth')
    ax.plot(model_count_np[i][0:100], color='red', label='prediction')

ax.legend()
axes[1].set_title("pred count")

# %% [markdown]
# visualize dataset

# %%
fig_limit = 49

fig, axes = plt.subplots(nrows=math.ceil(math.sqrt(fig_limit)), ncols=math.ceil(math.sqrt(fig_limit)), figsize=(64,64))
axes = axes.flatten()
i = 0
stop = False
for batch in neighs_batch_all:
    for g in batch.G:
        # print(g)
        ax = axes[i]
        node_color = ['blue' if g.nodes[n]['node_feature'].cpu().item()==0 else 'red' for n in g.nodes]
        nx.draw(g, pos = nx.kamada_kawai_layout(g), ax=ax, with_labels= False, node_color= node_color)
        i += 1
        if i >= fig_limit:
            stop = True
            break
    if stop:
        break

plt.suptitle(str(args.dataset_name), fontsize=64)

# %% [markdown]
# tests

# %%
# graph = query
# colors = ["red" if graph.nodes[node]['anchor']==1 else "blue" for node in graph.nodes()]
# pos = nx.kamada_kawai_layout(graph)
# nx.draw(graph, with_labels=True, node_color=colors, pos=pos)

# %% [markdown]
# # Visulize dns & sps conv params
# 
# do it only the conv is using TCONV

# %%
model.state_dict().keys()

# %%
dns_params = dict()
sps_params = dict()

if args.conv_type == 'TCONV':
    tconv_state_dict = model.state_dict()
    for key,value in tconv_state_dict.items():
        cmps = key.split('.')
        if len(cmps)<=4:
            continue
        if cmps[0] == 'emb_model' or cmps[0] == 'emb_model_query':
            if cmps[4] == 'bias':
                continue
            if cmps[3] == 'lin_dns':
                dns_params[key] = value.cpu().detach().numpy()
            if cmps[3] == 'lin_sps':
                sps_params[key] = value.cpu().detach().numpy()
    
    dns_params = [(k,v) for k,v in dns_params.items()]
    sps_params = [(k,v) for k,v in sps_params.items()]
else:
    pass


# %%
nrows = max(len(dns_params), len(sps_params))

fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(25,25*3))
ax = axes

for row in range(nrows):
    pic1 = dns_params[row][1]
    pic2 = sps_params[row][1]

    vmin = -0.3
    vmax = 0.3
    pcm = ax[row][0].pcolormesh(pic1, cmap= 'RdBu_r', norm= plt.Normalize(vmin=vmin, vmax=vmax))
    # fig.colorbar(pcm, ax=ax[row][0])
    ax[row][0].set_title(dns_params[row][0])
    pcm = ax[row][1].pcolormesh(pic2, cmap= 'RdBu_r', norm= plt.Normalize(vmin=vmin, vmax=vmax))
    # fig.colorbar(pcm, ax=ax[row][1])
    ax[row][1].set_title(sps_params[row][0])
    diff = pic1-pic2
    pcm = ax[row][2].pcolormesh(diff, cmap= 'RdBu_r', norm= plt.Normalize(vmin=vmin, vmax=vmax))
    # fig.colorbar(pcm, ax=ax[row][2])
    ax[row][2].set_title("diff")

# %%
