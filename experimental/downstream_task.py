import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import datetime
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


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
def test_acc_with_pred_count(count_path: str, test_mask: torch.Tensor, mlp: torch.nn.Module, y_test: torch.Tensor) -> float:
    # load the predicted count
    x_pred = pd.read_csv(count_path, index_col=0)
    # convert the count to torch tensor
    x_pred = torch.tensor(x_pred.values, dtype=torch.float).to(device)
    # extract the test set
    x_pred_test = x_pred[test_mask]
    # test the mlp model
    with torch.no_grad():
        logits = mlp(x_pred_test)
        pred = logits.max(1)[1]
        acc = pred.eq(y_test).sum().item() / y_test.size(0)
    print("test acc with predicted count: {}".format(acc))
    return acc


if __name__ == "__main__":
    dataset_name = "MUTAG"
    print("dataset: {}".format(dataset_name))

    dataset = TUDataset(root="data/{}".format(dataset_name), name=dataset_name)
    num_classes = dataset.num_classes

    # get the groundtruth count of standard queries in each graph
    query_ids = gen_query_ids(query_size= [3,4,5])
    train_workload = Workload(dataset, 'data/'+dataset_name, hetero_graph=True)
    if train_workload.exist_groundtruth(query_ids=query_ids):
        train_workload.canonical_count_truth = train_workload.load_groundtruth(query_ids=query_ids)
    else:
        train_workload.canonical_count_truth = train_workload.compute_groundtruth(query_ids= query_ids, save_to_file= True)
    train_workload.generate_pipeline_datasets(depth_neigh=4, neighborhood_transform=None) # depth can be set to any number larger than 3, won't matter

    # generate train mask and test mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_mask = torch.zeros(len(dataset), dtype=torch.bool)
    train_mask[:int(len(dataset) * 0.5)] = True
    train_mask = train_mask[torch.randperm(len(train_mask))] # shuffle the train_mask
    test_mask = ~train_mask

    # use the groundtruth count of standard quries to train the model
    x_truth = train_workload.gossip_dataset.aggregate_neighborhood_count(train_workload.canonical_count_truth).to(device) # shape (num_graphs, num_queries)
    label = dataset.data.y.to(device)

    x_truth_train = x_truth[train_mask]
    y_train = label[train_mask]
    x_truth_test = x_truth[test_mask]
    y_test = label[test_mask]

    # define the model
    mlp = MLP(input_dim= x_truth.shape[1], hidden_dim= 128, output_dim= num_classes).to(device)

    # train the mlp model
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for epoch in range(100):
        # train
        acc_loss = 0
        for x,y in DataLoader(list(zip(x_truth_train, y_train)), batch_size= 32):
            optimizer.zero_grad()
            logits = mlp(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
        # test
        with torch.no_grad():
            logits = mlp(x_truth_test)
            pred = logits.max(1)[1]
            acc = pred.eq(y_test).sum().item() / y_test.size(0)
        
        print("epoch: {}, loss: {}, acc: {}".format(epoch, acc_loss, acc))

    # print the accuracy of the model on the test set
    with torch.no_grad():
        logits = mlp(x_truth_test)
        pred = logits.max(1)[1]
        acc = pred.eq(y_test).sum().item() / y_test.size(0)
    print("test acc with ground truth count: {}".format(acc))
    
    # load the predicted count of standard quries to test the model
    count_paths = {
        'DeSCo': 'results/raw/graph_level/DeSCo/SAGE_Trival/sage_345_syn_qs_trival_epo300_{}.csv'.format(dataset_name), 
        'MOTIVO': 'results/raw/graph_level/MOTIVO/{}_atlas_counts_summary_345.csv'.format(dataset_name),
        'DIMNet': 'results/raw/graph_level/DIAMNet/GIN_DIAMNet_345_syn_qs_epo300_{}.csv'.format(dataset_name),
        'LRP': 'results/raw/graph_level/LRP/LRP_345_synXL_qs_epo50_{}.csv'.format(dataset_name),
        }
    for model_name, path in count_paths.items():
        print('test with model: {}'.format(model_name))
        test_acc_with_pred_count(path, test_mask, mlp, y_test)

    print('done')
