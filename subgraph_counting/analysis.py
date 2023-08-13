import os
import sys

import pickle
import networkx as nx
import pandas as pd
import numpy as np
from pandas import DataFrame
from numpy import ndarray

from subgraph_counting.config import parse_gossip, parse_neighborhood, parse_optimizer
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (
    GossipCountingModel,
    NeighborhoodCountingModel,
)
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.workload import Workload


def norm_mse(pred: ndarray, truth: ndarray, groupby: list[list] = None) -> list[float]:
    """
    Compute the normalized MSE for each group of queries.
    Args:
        pred: the predicted counts of queries, shape (num_graphs, num_queries)
        truth: the groundtruth counts of queries, shape (num_graphs, num_queries)
        groupby: a list of lists, each list contains the indices of queries in the same group. If None, all queries are in the same group.
    """
    if groupby is None:
        groupby = [list(range(pred.shape[1]))]

    pred = pred.astype(np.float64)
    truth = truth.astype(np.float64)
    norm_mse_list = []
    for group in groupby:
        mse = np.mean(((pred[:, group] - truth[:, group]) ** 2))
        norm = np.var(truth[:, group])
        norm_mse = mse / norm
        print("mean_norm_mse: ", norm_mse)
        norm_mse_list.append(norm_mse)

    return norm_mse_list


def mse(pred: ndarray, truth: ndarray, groupby: list[list] = None) -> list[float]:
    """
    Compute the normalized MSE for each group of queries.
    Args:
        pred: the predicted counts of queries, shape (num_graphs, num_queries)
        truth: the groundtruth counts of queries, shape (num_graphs, num_queries)
        groupby: a list of lists, each list contains the indices of queries in the same group. If None, all queries are in the same group.
    """
    if groupby is None:
        groupby = [list(range(pred.shape[1]))]

    pred = pred.astype(np.float64)
    truth = truth.astype(np.float64)
    mse_list = []
    for group in groupby:
        mse = np.mean(((pred[:, group] - truth[:, group]) ** 2))
        print("mean_mse: ", mse)
        mse_list.append(mse)

    return mse_list


def mae(pred: ndarray, truth: ndarray, groupby: list[list]) -> list[float]:
    """
    Compute the normalized MSE for each group of queries.
    Args:
        pred: the predicted counts of queries, shape (num_graphs, num_queries)
        truth: the groundtruth counts of queries, shape (num_graphs, num_queries)
        groupby: a list of lists, each list contains the indices of queries in the same group
    """
    mae_list = []

    for group in groupby:
        mae = np.mean(np.abs(pred[:, group] - truth[:, group]))
        print("mean_mae: ", mae)
        mae_list.append(mae)

    return mae_list


def graphlet_counting_analysis(
    dataset_name: str,
    data_path: str,
    raw_output_path: str,
    groupby: list[list],
    save_to_file: bool = False,
):
    """
    Args:
        dataset_name: the name of the dataset
        data_path: the path of the dataset
        raw_output_path: the path of the raw output of graphlet counting
        groupby: a list of lists, each list contains the indices of queries in the same group
        save_to_file: whether to save the results to file
    """
    raise NotImplementedError


if __name__ == "__main__":
    # read csv
    count_pred = pd.read_csv(graphlet_path, index_col=0).to_numpy()

    # get the groundtruth count of standard queries in each graph
    # get queries
    baseline_query_path = "data/MUTAG/baseline_patterns"
    num_node_label = 7
    nx_queries = []
    for file in os.listdir(baseline_query_path):
        if file.endswith(".gml"):
            nx_graph = nx.read_gml(
                os.path.join(baseline_query_path, file), label="id"
            ).to_undirected()
            for node in nx_graph.nodes:
                # label = nx_graph.nodes[node]['label']
                # nx_graph.nodes[node]['feat'] = [0.0 for i in range(num_node_label)]
                # nx_graph.nodes[node]['feat'][int(label)] = 1.0
                nx_graph.nodes[node]["feat"] = [0.0]
            nx_queries.append(nx_graph)
    query_ids = None

    # get target
    dataset_name = "MUTAG_test"
    dataset = load_data(dataset_name)
    train_workload = Workload(
        dataset, "data/" + dataset_name, hetero_graph=True, node_feat_len=7
    )
    if train_workload.exist_groundtruth(query_ids=query_ids, queries=nx_queries):
        train_workload.canonical_count_truth = train_workload.load_groundtruth(
            query_ids=query_ids, queries=nx_queries
        )
    else:
        train_workload.canonical_count_truth = train_workload.compute_groundtruth(
            query_ids=query_ids, queries=nx_queries, num_workers=4, save_to_file=False
        )
    train_workload.generate_pipeline_datasets(
        depth_neigh=4, neighborhood_transform=None
    )

    # use the groundtruth count of standard quries to train the model
    count_truth = train_workload.gossip_dataset.aggregate_neighborhood_count(
        train_workload.canonical_count_truth
    ).numpy()  # shape (num_graphs, num_queries)

    print("count_truth")
    print(count_truth)
    print("count_pred")
    print(count_pred)

    # result analysis
    rmse = np.sqrt(np.mean((count_pred - count_truth) ** 2))

    mae = np.mean(np.abs(count_pred - count_truth))

    print("MSE: ", mse)

    print("RMSE: ", rmse)
    print("MAE: ", mae)

    print("done")
