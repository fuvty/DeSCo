import os
import sys

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

from subgraph_counting.config import parse_gossip, parse_neighborhood, parse_optimizer
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (
    GossipCountingModel,
    NeighborhoodCountingModel,
)
from subgraph_counting.lightning_data import LightningDataLoader
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat
from subgraph_counting.workload import Workload
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.analysis import norm_mse, mae


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
        nx_queries = [nx.graph_atlas(i) for i in atlas_query_ids]
        if args_neighborhood.use_node_feature:
            # TODO: remove this in future implementations
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
    load_data_transform = [T.ToUndirected()]
    if args_neighborhood.zero_node_feat:
        load_data_transform.append(ZeroNodeFeat())

    # neighborhood transformation
    neighborhood_transform = ToTconvHetero() if args_neighborhood.use_tconv else None
    assert args_neighborhood.use_hetero if args_neighborhood.use_tconv else True

    if train_neighborhood or train_gossip:
        # define training workload
        train_dataset_name = args_opt.train_dataset
        train_dataset = load_data(
            train_dataset_name, transform=load_data_transform
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

        # define validation workload
        valid_dataset_name = args_opt.valid_dataset
        valid_dataset = load_data(valid_dataset_name, transform=load_data_transform)
        valid_workload = Workload(
            valid_dataset,
            "data/" + valid_dataset_name,
            hetero_graph=True,
            node_feat_len=args_neighborhood.input_dim
            if args_neighborhood.use_node_feature
            else -1,
        )
        if valid_workload.exist_groundtruth(query_ids=query_ids, queries=nx_queries):
            valid_workload.canonical_count_truth = valid_workload.load_groundtruth(
                query_ids=query_ids, queries=nx_queries
            )
        else:
            valid_workload.canonical_count_truth = valid_workload.compute_groundtruth(
                query_ids=query_ids,
                queries=nx_queries,
                num_workers=args_opt.num_cpu,
                save_to_file=True,
            )
        valid_workload.generate_pipeline_datasets(
            depth_neigh=args_neighborhood.depth,
            neighborhood_transform=neighborhood_transform,
        )  # generate pipeline dataset, including neighborhood dataset and gossip dataset

    # define testing workload
    test_dataset_name = args_opt.test_dataset
    test_dataset = load_data(test_dataset_name, transform=load_data_transform)
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

    # define devices
    if type(args_opt.gpu) == int:  # single gpu
        devices = [args_opt.gpu]
    elif type(args_opt.gpu) == list:  # multiple gpus
        devices = args_opt.gpu
    else:
        Warning("gpu is not specified, using auto mode")
        devices = ["auto"]
    device = devices[0]

    ########### begin neighborhood counting ###########

    # define neighborhood counting dataset
    neighborhood_dataloader = LightningDataLoader(
        train_dataset=train_workload.neighborhood_dataset
        if (train_neighborhood or train_gossip)
        else None,
        val_dataset=valid_workload.neighborhood_dataset if train_neighborhood else None,
        test_dataset=test_workload.neighborhood_dataset,
        batch_size=args_neighborhood.batch_size,
        num_workers=args_opt.num_cpu,
        shuffle=False,
    )

    # define neighborhood counting model
    neighborhood_trainer = pl.Trainer(
        max_epochs=args_neighborhood.epoch_num,
        accelerator="gpu",
        devices=[device],  # use only one gpu except for training
        default_root_dir=args_neighborhood.model_path,
        auto_lr_find=args_neighborhood.tune_lr,
        auto_scale_batch_size=args_neighborhood.tune_bs,
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
        if args_neighborhood.tune_lr or args_neighborhood.tune_bs:
            neighborhood_trainer.tune(
                model=neighborhood_model, datamodule=neighborhood_dataloader
            )
        # multi-gpu training
        if len(devices) > 1:
            neighborhood_multigpu_trainer = pl.Trainer(
                max_epochs=args_neighborhood.epoch_num,
                accelerator="gpu",
                devices=devices,  # use multiple gpus for training
                default_root_dir=args_neighborhood.model_path,
                gpus=devices,
                strategy="ddp",
            )
            neighborhood_multigpu_trainer.fit(
                model=neighborhood_model,
                datamodule=neighborhood_dataloader,
            )
        else:
            neighborhood_trainer.fit(
                model=neighborhood_model,
                datamodule=neighborhood_dataloader,
            )
    # test neighborhood counting model
    neighborhood_trainer.test(
        model=neighborhood_model, datamodule=neighborhood_dataloader
    )

    ########### begin gossip counting ###########
    skip_gossip = not (train_gossip or test_gossip)
    # apply neighborhood count output to gossip dataset
    if train_gossip:
        neighborhood_count_train = torch.cat(
            neighborhood_trainer.predict(
                neighborhood_model, neighborhood_dataloader.train_dataloader()
            ),
            dim=0,
        )  # size = (#neighborhood, #queries)
        train_workload.apply_neighborhood_count(neighborhood_count_train)
    if test_gossip:
        neighborhood_count_test = torch.cat(
            neighborhood_trainer.predict(
                neighborhood_model, neighborhood_dataloader.test_dataloader()
            ),
            dim=0,
        )
        test_workload.apply_neighborhood_count(neighborhood_count_test)

    # define gossip counting dataset
    if not skip_gossip:
        gossip_dataloader = LightningDataLoader(
            train_dataset=train_workload.gossip_dataset if train_gossip else None,
            val_dataset=valid_workload.gossip_dataset if train_gossip else None,
            test_dataset=test_workload.gossip_dataset,
            batch_size=args_gossip.batch_size,
            num_workers=args_opt.num_cpu,
            shuffle=False,
        )

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
            devices=[device],  # use only one gpu except for training
            default_root_dir=args_gossip.model_path,
            detect_anomaly=True,
            auto_lr_find=args_gossip.tune_lr,
            auto_scale_batch_size="power" if args_gossip.tune_bs else None,
        )

    # train gossip model
    if train_gossip:
        if args_gossip.tune_lr or args_gossip.tune_bs:
            gossip_trainer.tune(gossip_model, gossip_dataloader)
        if len(devices) > 1:
            gossip_multigpu_trainer = pl.Trainer(
                max_epochs=args_gossip.epoch_num,
                accelerator="gpu",
                devices=devices,  # use multiple gpus for training
                default_root_dir=args_gossip.model_path,
                strategy="ddp",
            )
            gossip_multigpu_trainer.fit(
                model=gossip_model,
                datamodule=gossip_dataloader,
            )
        else:
            gossip_trainer.fit(
                model=gossip_model,
                datamodule=gossip_dataloader,
            )
    elif test_gossip:
        gossip_trainer.test(gossip_model, datamodule=gossip_dataloader)

    ########### output prediction results ###########
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
        neighborhood_trainer.predict(
            neighborhood_model, neighborhood_dataloader.test_dataloader()
        ),
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
            gossip_trainer.predict(gossip_model, gossip_dataloader.test_dataloader()),
            dim=0,
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

    ########### analyze the output data ###########
    # group the results by query graph size
    query_size_dict = {i: len(g) for i, g in enumerate(nx_queries)}
    size_order_dict = {
        size: i for i, size in enumerate(sorted(set(query_size_dict.values())))
    }
    groupby_list = [[] for _ in range(len(size_order_dict))]
    for i in query_size_dict.keys():
        groupby_list[size_order_dict[query_size_dict[i]]].append(i)

    # analyze the graphlet count for neighborhood counting
    truth_graphlet = test_workload.gossip_dataset.aggregate_neighborhood_count(
        test_workload.canonical_count_truth
    ).numpy()  # shape (num_graphs, num_queries)
    pred_graphlet_neighborhood = (
        torch.round(F.relu(graphlet_neighborhood_count_test)).detach().cpu().numpy()
    )
    norm_mse_neighborhood = norm_mse(
        pred=pred_graphlet_neighborhood, truth=truth_graphlet, groupby=groupby_list
    )
    mae_neighborhood = mae(
        pred=pred_graphlet_neighborhood, truth=truth_graphlet, groupby=groupby_list
    )
    print("graphlet_norm_mse_neighborhood: {}".format(norm_mse_neighborhood))
    print("graphlet_mae_neighborhood: {}".format(mae_neighborhood))

    # analyze the graphlet count for gossip counting
    if not skip_gossip:
        pred_graphlet_gossip = (
            torch.round(F.relu(graphlet_gossip_count_test)).detach().cpu().numpy()
        )
        norm_mse_gossip = norm_mse(
            pred=pred_graphlet_gossip, truth=truth_graphlet, groupby=groupby_list
        )
        mae_gossip = mae(
            pred=pred_graphlet_gossip, truth=truth_graphlet, groupby=groupby_list
        )
        print("graphlet_norm_mse_gossip: {}".format(norm_mse_gossip))
        print("graphlet_mae_gossip: {}".format(mae_gossip))

    # save the results
    file_name = "analyze_results_{}.txt".format(args_opt.test_dataset)
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write("graphlet_norm_mse_neighborhood: {}\n".format(norm_mse_neighborhood))
        f.write("graphlet_mae_neighborhood: {}\n".format(mae_neighborhood))
        if not skip_gossip:
            f.write("graphlet_norm_mse_gossip: {}\n".format(norm_mse_gossip))
            f.write("graphlet_mae_gossip: {}\n".format(mae_gossip))

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
