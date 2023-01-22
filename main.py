import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import datetime
from typing import List, Optional, Tuple, Union
import pickle
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
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
from subgraph_counting.workload import Workload
from subgraph_counting.utils import add_node_feat_to_networkx


def main(
    args_neighborhood,
    args_gossip,
    args_opt,
    train_neighborhood: bool = True,
    train_gossip: bool = True,
    test_gossip: bool = True,
    neighborhood_checkpoint=None,
    gossip_checkpoint=None,
    nx_queries: List[nx.Graph] = None,
    atlas_query_ids: List[int] = None,
    output_dir: str = "results/raw",
):
    """
    train the model and test accorrding to the config
    """

    # define queries by atlas ids or networkx graphs
    if nx_queries is None and atlas_query_ids is not None:
        if args_neighborhood.use_node_feature:
            # TODO: remove this in future implementations
            nx_queries = [nx.graph_atlas(i) for i in atlas_query_ids]
            nx_queries_with_node_feat = []
            for query in nx_queries:
                nx_queries_with_node_feat.extend(
                    add_node_feat_to_networkx(
                        query,
                        [t for t in np.eye(args_neighborhood.input_dim).tolist()],
                        "feat",
                    )
                )
                nx_queries = nx_queries_with_node_feat
            query_ids = None
            print("define queries with atlas ids:", query_ids)
            print("query_ids set to None because node features are used")
        else:
            query_ids = atlas_query_ids
            print("define queries with atlas ids:", query_ids)
    elif nx_queries is not None and atlas_query_ids is None:
        query_ids = None
        print("define queries with nx graphs, number of query is", len(nx_queries))
        print("length of nx_queries are: ", [len(q) for q in nx_queries])
        print("query_ids set to None")
    elif nx_queries is not None and atlas_query_ids is not None:
        raise ValueError("nx_queries and atlas_query_ids cannot be both empty")
    else:
        raise ValueError("nx_queries and atlas_query_ids cannot be both None")

    # define pre-transform
    zero_node_feat_transform = (
        ZeroNodeFeat() if args_neighborhood.zero_node_feat else None
    )

    # neighborhood transformation
    neighborhood_transform = ToTconvHetero() if args_neighborhood.use_tconv else None
    assert args_neighborhood.use_hetero if args_neighborhood.use_tconv else True

    # define training workload
    if train_neighborhood or train_gossip:
        train_dataset_name = args_opt.train_dataset
        train_dataset = load_data(
            train_dataset_name, transform=zero_node_feat_transform
        )  # TODO: add valid set mask support
        train_workload = Workload(
            train_dataset,
            "data/" + train_dataset_name,
            hetero_graph=True,
            node_feat_len=args_neighborhood.input_dim
            if args_neighborhood.use_node_feature
            else -1,
        )
        if train_workload.exist_groundtruth(query_ids=query_ids, queries=nx_queries):
            train_workload.canonical_count_truth = train_workload.load_groundtruth(
                query_ids=query_ids, queries=nx_queries
            )
        else:
            train_workload.canonical_count_truth = train_workload.compute_groundtruth(
                query_ids=query_ids,
                queries=nx_queries,
                num_workers=args_opt.num_cpu,
                save_to_file=True,
            )
        train_workload.generate_pipeline_datasets(
            depth_neigh=args_neighborhood.depth,
            neighborhood_transform=neighborhood_transform,
        )  # generate pipeline dataset, including neighborhood dataset and gossip dataset

    # define testing workload
    test_dataset_name = args_opt.test_dataset
    test_dataset = load_data(test_dataset_name, transform=zero_node_feat_transform)
    test_workload = Workload(
        test_dataset,
        "data/" + test_dataset_name,
        hetero_graph=True,
        node_feat_len=args_neighborhood.input_dim
        if args_neighborhood.use_node_feature
        else -1,
    )
    if test_workload.exist_groundtruth(query_ids=query_ids, queries=nx_queries):
        test_workload.canonical_count_truth = test_workload.load_groundtruth(
            query_ids=query_ids, queries=nx_queries
        )
    else:
        test_workload.canonical_count_truth = test_workload.compute_groundtruth(
            query_ids=query_ids,
            queries=nx_queries,
            num_workers=args_opt.num_cpu,
            save_to_file=True,
        )  # compute ground truth if not any
    test_workload.generate_pipeline_datasets(
        depth_neigh=args_neighborhood.depth,
        neighborhood_transform=neighborhood_transform,
    )  # generate pipeline dataset, including neighborhood dataset and gossip dataset

    ########### begin neighborhood counting ###########
    # define neighborhood counting dataset
    if train_neighborhood or train_gossip:
        neighborhood_train_dataloader = DataLoader(
            train_workload.neighborhood_dataset,
            batch_size=args_opt.neighborhood_batch_size,
            shuffle=False,
            num_workers=args_opt.num_cpu,
        )
    neighborhood_test_dataloader = DataLoader(
        test_workload.neighborhood_dataset,
        batch_size=args_opt.neighborhood_batch_size,
        shuffle=False,
        num_workers=args_opt.num_cpu,
    )

    # define neighborhood counting model
    neighborhood_trainer = pl.Trainer(
        max_epochs=args_neighborhood.epoch_num,
        accelerator="gpu",
        devices=[args_opt.gpu],
        default_root_dir=args_neighborhood.model_path,
    )

    if train_neighborhood:
        neighborhood_model = NeighborhoodCountingModel(
            input_dim=args_neighborhood.input_dim,
            hidden_dim=args_neighborhood.hidden_dim,
            args=args_neighborhood,
        )
        neighborhood_model = neighborhood_model.to_hetero_old(
            tconv_target=args_neighborhood.use_tconv,
            tconv_query=args_neighborhood.use_tconv,
        )
    else:
        assert neighborhood_checkpoint is not None
        neighborhood_model = NeighborhoodCountingModel.load_from_checkpoint(
            neighborhood_checkpoint
        )  # to hetero is automatically done upon loading
    neighborhood_model.set_queries(
        query_ids=query_ids, queries=nx_queries, transform=neighborhood_transform
    )

    # train neighborhood model
    if train_neighborhood:
        neighborhood_trainer.fit(
            model=neighborhood_model,
            train_dataloaders=neighborhood_train_dataloader,
            val_dataloaders=neighborhood_test_dataloader,
        )

    # test neighborhood counting model
    neighborhood_trainer.test(
        neighborhood_model, dataloaders=neighborhood_test_dataloader
    )

    ########### begin gossip counting ###########
    skip_gossip = not (train_gossip or test_gossip)
    # apply neighborhood count output to gossip dataset
    if train_gossip:
        neighborhood_count_train = torch.cat(
            neighborhood_trainer.predict(
                neighborhood_model, neighborhood_train_dataloader
            ),
            dim=0,
        )  # size = (#neighborhood, #queries)
        train_workload.apply_neighborhood_count(neighborhood_count_train)
    if test_gossip:
        neighborhood_count_test = torch.cat(
            neighborhood_trainer.predict(
                neighborhood_model, neighborhood_test_dataloader
            ),
            dim=0,
        )
        test_workload.apply_neighborhood_count(neighborhood_count_test)
        # define gossip counting dataset
        gossip_test_dataloader = DataLoader(test_workload.gossip_dataset)

    # define gossip counting model
    input_dim = 1
    args_gossip.use_hetero = False
    if train_gossip:
        gossip_model = GossipCountingModel(
            input_dim,
            args_gossip.hidden_dim,
            args_gossip,
            emb_channels=args_neighborhood.hidden_dim,
            input_pattern_emb=True,
        )
    elif test_gossip:
        assert gossip_checkpoint is not None
        gossip_model = GossipCountingModel.load_from_checkpoint(gossip_checkpoint)
    else:
        print("No gossip training or testing is specified, skip gossip counting.")

    if not skip_gossip:
        gossip_model.set_query_emb(neighborhood_model.get_query_emb())

        gossip_trainer = pl.Trainer(
            max_epochs=args_gossip.epoch_num,
            accelerator="gpu",
            devices=[args_opt.gpu],
            default_root_dir=args_gossip.model_path,
            detect_anomaly=True,
        )

    # train gossip model
    if train_gossip:
        gossip_train_dataloader = DataLoader(train_workload.gossip_dataset)
        gossip_trainer.fit(
            model=gossip_model,
            train_dataloaders=gossip_train_dataloader,
            val_dataloaders=gossip_test_dataloader,
        )
    elif test_gossip:
        gossip_trainer.test(gossip_model, dataloaders=gossip_test_dataloader)

    ########### output graphlet results ###########
    # configurations
    file_name = "config_{}.txt".format(args_opt.test_dataset)
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write("args_opt: \n")
        f.write(str(args_opt))
        f.write("\nargs_neighborhood:\n")
        f.write(str(args_neighborhood))
        f.write("\nargs_gossip:\n")
        f.write(str(args_gossip))
        f.write("\ntime:\n")
        f.write(str(datetime.datetime.now()))

    neighborhood_count_test = torch.cat(
        neighborhood_trainer.predict(neighborhood_model, neighborhood_test_dataloader),
        dim=0,
    )
    graphlet_neighborhood_count_test = (
        test_workload.neighborhood_dataset.aggregate_neighborhood_count(
            neighborhood_count_test
        )
    )  # user can get the graphlet count of each graph in this way

    # graphlet count after neighborhood counting
    file_name = "neighborhood_graphlet_{}.csv".format(args_opt.test_dataset)
    pd.DataFrame(
        torch.round(F.relu(graphlet_neighborhood_count_test)).detach().cpu().numpy()
    ).to_csv(
        os.path.join(output_dir, file_name)
    )  # save the inferenced results to csv file

    # graphlet count after gossip counting
    if not skip_gossip:
        file_name = "gossip_graphlet_{}.csv".format(args_opt.test_dataset)
        gossip_count_test = torch.cat(
            gossip_trainer.predict(gossip_model, gossip_test_dataloader), dim=0
        )
        graphlet_gossip_count_test = (
            test_workload.gossip_dataset.aggregate_neighborhood_count(gossip_count_test)
        )  # user can get the graphlet count of each graph in this way
        pd.DataFrame(
            torch.round(F.relu(graphlet_gossip_count_test)).detach().cpu().numpy()
        ).to_csv(
            os.path.join(output_dir, file_name)
        )  # save the inferenced results to csv file

    # node level count after neighborhood counting
    file_name = "neighborhood_node_{}".format(args_opt.test_dataset)
    pd.DataFrame(neighborhood_count_test.detach().cpu().numpy()).to_csv(
        os.path.join(output_dir, file_name + "_results.csv")
    )  # save the inferenced results to csv file
    pd.DataFrame(test_workload.neighborhood_dataset.nx_neighs_index).to_csv(
        os.path.join(output_dir, file_name + "_index.csv")
    )  # save the inferenced results to csv file

    # node level count after gossip counting
    file_name = "gossip_node_{}".format(args_opt.test_dataset)
    if not skip_gossip:
        pd.DataFrame(gossip_count_test.detach().cpu().numpy()).to_csv(
            os.path.join(output_dir, file_name + "_results.csv")
        )  # save the inferenced results to csv file

    # save the test networkx graph
    file_name = "test_nxgraph_{}.pk".format(args_opt.test_dataset)
    with open(os.path.join(output_dir, file_name), "wb") as f:
        pickle.dump(test_workload.to_networkx(), f)

    print("done")


if __name__ == "__main__":
    # load parameters
    parser = argparse.ArgumentParser(description="DeSCo argument parser")

    # define optimizer arguments
    _optimizer_actions = parse_optimizer(parser)

    # define neighborhood counting model arguments
    _neighbor_actions = parse_neighborhood(parser)

    # define gossip counting model arguments
    _gossip_actions = parse_gossip(parser)

    # assign the args to args_neighborhood, args_gossip, and args_opt without the prefix 'neigh_' and 'gossip_'
    args = parser.parse_args()
    print(args)
    args_neighborhood = argparse.Namespace()
    args_gossip = argparse.Namespace()
    args_opt = argparse.Namespace()
    for action in _optimizer_actions:
        setattr(args_opt, action.dest, getattr(args, action.dest))
    for action in _neighbor_actions:
        prefix = "neigh_"
        setattr(
            args_neighborhood,
            action.dest[len(prefix) :]
            if action.dest.startswith(prefix)
            else action.dest,
            getattr(args, action.dest),
        )
    for action in _gossip_actions:
        prefix = "gossip_"
        setattr(
            args_gossip,
            action.dest[len(prefix) :]
            if action.dest.startswith(prefix)
            else action.dest,
            getattr(args, action.dest),
        )

    # noqa: the following restrictions are added because of the limited implemented senarios
    assert args_neighborhood.use_hetero == True

    # noqa: if need to load from checkpoint, please specify the checkpoint path
    neighborhood_checkpoint = args_opt.neigh_checkpoint
    gossip_checkpoint = args_opt.gossip_checkpoint

    # define the query graphs
    query_ids = gen_query_ids(query_size=[3, 4, 5])
    # query_ids = [6]
    nx_queries = [nx.graph_atlas(i) for i in query_ids]
    if args_neighborhood.use_node_feature:
        nx_queries_with_node_feat = []
        for query in nx_queries:
            nx_queries_with_node_feat.extend(
                add_node_feat_to_networkx(
                    query,
                    [t for t in np.eye(args_neighborhood.input_dim).tolist()],
                    "feat",
                )
            )
            nx_queries = nx_queries_with_node_feat
    query_ids = None

    # load gml queries from file
    # num_node_label = args_neighborhood.input_dim
    # baseline_query_path = 'data/MUTAG/baseline_patterns'

    # iterate through all files ending with gml in the directory
    # nx_queries = []
    # for file in os.listdir(baseline_query_path):
    #     if file.endswith(".gml"):
    #         nx_graph = nx.read_gml(os.path.join(baseline_query_path, file), label='id').to_undirected()
    #         for node in nx_graph.nodes:
    #             label = nx_graph.nodes[node]['label']
    #             nx_graph.nodes[node]['feat'] = [0.0 for i in range(num_node_label)]
    #             nx_graph.nodes[node]['feat'][int(label)] = 1.0
    #             # nx_graph.nodes[node]['feat'] = [0.0]
    #         nx_queries.append(nx_graph)

    query_ids = None

    # define the output directory
    if args_opt.output_dir is None:
        output_dir = "results/kdd23/raw"
        time = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
        output_dir = os.path.join(output_dir, time)
    else:
        output_dir = args_opt.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(
        args_neighborhood,
        args_gossip,
        args_opt,
        train_neighborhood=args_opt.train_neigh,
        train_gossip=args_opt.train_gossip,
        test_gossip=args_opt.test_gossip,
        neighborhood_checkpoint=neighborhood_checkpoint,
        gossip_checkpoint=gossip_checkpoint,
        nx_queries=nx_queries,
        atlas_query_ids=query_ids,
        output_dir=output_dir,
    )
