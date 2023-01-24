import argparse
import networkx as nx
from subgraph_counting.workload import Workload
from subgraph_counting.data import load_data, gen_query_ids
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
import torch_geometric.transforms as T

if __name__ == "__main__":
    # *************** define the arguments *************** #
    count_queries = False

    train_dataset_name = "syn_64"

    args_neighborhood = argparse.Namespace()
    args_neighborhood.depth = 4
    args_neighborhood.use_node_feature = False
    args_neighborhood.input_dim = -1
    args_neighborhood.use_tconv = True
    args_neighborhood.use_hetero = True
    args_neighborhood.zero_node_feat = False

    num_cpu = 16

    # define the query graphs
    query_ids = gen_query_ids(query_size=[3, 4, 5])
    # query_ids = [6]
    nx_queries = [nx.graph_atlas(i) for i in query_ids]
    if args_neighborhood.use_node_feature:
        raise NotImplementedError
    query_ids = None

    # *************** generate dataset *************** #
    print("define queries with nx graphs, number of query is", len(nx_queries))
    print("length of nx_queries are: ", [len(q) for q in nx_queries])
    print("query_ids set to None")

    # define pre-transform
    load_data_transform = [T.ToUndirected()]
    if args_neighborhood.zero_node_feat:
        load_data_transform.append(ZeroNodeFeat())

    train_dataset = load_data(train_dataset_name, transform=load_data_transform)

    neighborhood_transform = ToTconvHetero() if args_neighborhood.use_tconv else None
    assert args_neighborhood.use_hetero if args_neighborhood.use_tconv else True

    train_workload = Workload(
        train_dataset,
        "data/" + train_dataset_name,
        hetero_graph=True,
        node_feat_len=args_neighborhood.input_dim
        if args_neighborhood.use_node_feature
        else -1,
    )

    if count_queries:
        if train_workload.exist_groundtruth(query_ids=query_ids, queries=nx_queries):
            train_workload.canonical_count_truth = train_workload.load_groundtruth(
                query_ids=query_ids, queries=nx_queries
            )
        else:
            train_workload.canonical_count_truth = train_workload.compute_groundtruth(
                query_ids=query_ids,
                queries=nx_queries,
                num_workers=num_cpu,
                save_to_file=True,
            )

    train_workload.generate_pipeline_datasets(
        depth_neigh=args_neighborhood.depth,
        neighborhood_transform=neighborhood_transform,
    )  # generate pipeline dataset, including neighborhood dataset and gossip dataset
