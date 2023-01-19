import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import pickle
import networkx as nx
import pandas as pd
import numpy as np

from subgraph_counting.config import (parse_gossip, parse_neighborhood,
                                      parse_optimizer)
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (GossipCountingModel,
                                               NeighborhoodCountingModel)
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.workload import Workload

if __name__ == "__main__":
    graphlet_path = 'results/raw_results/neighborhood_SAGE_MUTAG_test_20221116_00:40:19.csv'

    # read csv
    count_pred = pd.read_csv(graphlet_path, index_col=0).to_numpy()


    # get the groundtruth count of standard queries in each graph
    # get queries
    baseline_query_path = 'data/MUTAG/baseline_patterns'
    num_node_label = 7
    nx_queries = []
    for file in os.listdir(baseline_query_path):
        if file.endswith(".gml"):
            nx_graph = nx.read_gml(os.path.join(baseline_query_path, file), label='id').to_undirected()
            for node in nx_graph.nodes:
                # label = nx_graph.nodes[node]['label']
                # nx_graph.nodes[node]['feat'] = [0.0 for i in range(num_node_label)]
                # nx_graph.nodes[node]['feat'][int(label)] = 1.0
                nx_graph.nodes[node]['feat'] = [0.0]
            nx_queries.append(nx_graph)
    query_ids = None

    # get target
    dataset_name = 'MUTAG_test'
    dataset = load_data(dataset_name)
    train_workload = Workload(dataset, 'data/'+dataset_name, hetero_graph=True, node_feat_len=7)
    if train_workload.exist_groundtruth(query_ids=query_ids, queries=nx_queries):
        train_workload.canonical_count_truth = train_workload.load_groundtruth(query_ids=query_ids, queries=nx_queries)
    else:
        train_workload.canonical_count_truth = train_workload.compute_groundtruth(query_ids= query_ids, queries=nx_queries, num_workers= 4, save_to_file= False)
    train_workload.generate_pipeline_datasets(depth_neigh=4, neighborhood_transform=None)

    # use the groundtruth count of standard quries to train the model
    count_truth = train_workload.gossip_dataset.aggregate_neighborhood_count(train_workload.canonical_count_truth).numpy() # shape (num_graphs, num_queries)

    print("count_truth")
    print(count_truth)
    print("count_pred")
    print(count_pred)

    # result analysis
    rmse = np.sqrt(np.mean((count_pred - count_truth)**2))
    mse = np.mean((count_pred - count_truth)**2)
    mae = np.mean(np.abs(count_pred - count_truth))

    print('MSE: ', mse)

    print('RMSE: ', rmse)
    print('MAE: ', mae)

    print('done')