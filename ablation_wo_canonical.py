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
from pytorch_lightning.callbacks import ModelCheckpoint
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
from subgraph_counting.workload import Workload_baseline
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.analysis import norm_mse, mae


def main(
    args_neighborhood,
    args_gossip,
    args_opt,
    train_neighborhood: bool = True,
    train_gossip: bool = True,
    neighborhood_checkpoint=None,
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
        train_dataset = load_data(train_dataset_name, transform=None)

        train_workload = Workload_baseline(
            train_dataset,
            "data/" + train_dataset_name,
            hetero_graph=False,
            node_feat_len=-1,
        )

        args.dataset = train_dataset_name
        args.dataset_name = train_dataset_name

        train_workload.get_canonical_count_truth(query_ids=None, queries=nx_queries)
        train_workload.canonical_to_graphlet_truth(
            canonical_count_truth=train_workload.canonical_count_truth
        )
        train_workload.generate_wo_canonical_dataset(transform=neighborhood_transform)

        # define validation workload
        val_dataset = train_dataset
        val_dataset_name = args_opt.valid_dataset
        val_workload = Workload_baseline(
            val_dataset,
            "data/" + val_dataset_name,
            hetero_graph=False,
            node_feat_len=-1,
        )

        args.dataset = val_dataset_name
        args.dataset_name = val_dataset_name

        val_workload.get_canonical_count_truth(query_ids=None, queries=nx_queries)
        val_workload.canonical_to_graphlet_truth(
            canonical_count_truth=val_workload.canonical_count_truth
        )
        val_workload.generate_wo_canonical_dataset(transform=neighborhood_transform)

    # define testing workload
    test_dataset_name = args_opt.test_dataset
    test_dataset = load_data(test_dataset_name, transform=None)

    test_workload = Workload_baseline(
        test_dataset,
        "data/" + test_dataset_name,
        hetero_graph=False,
        node_feat_len=-1,
    )
    args.dataset = test_dataset_name
    args.dataset_name = test_dataset_name
    test_workload.get_canonical_count_truth(query_ids=None, queries=nx_queries)
    test_workload.canonical_to_graphlet_truth(
        canonical_count_truth=test_workload.canonical_count_truth
    )
    test_workload.generate_wo_canonical_dataset(transform=neighborhood_transform)

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
        train_dataset=train_workload.wo_canonical_dataset
        if (train_neighborhood or train_gossip)
        else None,
        val_dataset=val_workload.wo_canonical_dataset
        if (train_neighborhood or train_gossip)
        else None,
        test_dataset=test_workload.wo_canonical_dataset,
        batch_size=args_neighborhood.batch_size,
        num_workers=args_opt.num_cpu,
        shuffle=False,
    )

    # define neighborhood counting model
    neighborhood_checkpoint_callback = ModelCheckpoint(
        monitor="neighborhood_counting_val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    neighborhood_trainer = pl.Trainer(
        max_epochs=args_neighborhood.epoch_num,
        accelerator="gpu",
        devices=[device],  # use only one gpu except for training
        default_root_dir=args_neighborhood.model_path,
        callbacks=[neighborhood_checkpoint_callback],
        auto_lr_find=args_neighborhood.tune_lr,
        auto_scale_batch_size=args_neighborhood.tune_bs,
    )

    if train_neighborhood and (neighborhood_checkpoint is None):
        neighborhood_model = NeighborhoodCountingModel(
            input_dim=args_neighborhood.input_dim,
            hidden_dim=args_neighborhood.hidden_dim,
            args=args_neighborhood,
        )
        neighborhood_model = neighborhood_model.to_hetero_wo_canonical(
            tconv_target=args_neighborhood.use_tconv,
            tconv_query=args_neighborhood.use_tconv,
        )
    else:
        assert neighborhood_checkpoint is not None
        print("loading neighborhood model from checkpoint: ", neighborhood_checkpoint)
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
                callbacks=[neighborhood_checkpoint_callback],
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
        neighborhood_best_model_path = neighborhood_checkpoint_callback.best_model_path
        print("best neighborhood model path: ", neighborhood_best_model_path)
        neighborhood_model = NeighborhoodCountingModel.load_from_checkpoint(
            neighborhood_best_model_path
        )
        neighborhood_model.set_queries(
            query_ids=query_ids, queries=nx_queries, transform=neighborhood_transform
        )

    # test neighborhood counting model
    neighborhood_trainer.test(
        model=neighborhood_model, datamodule=neighborhood_dataloader
    )

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

    count_pred = torch.cat(
        neighborhood_trainer.predict(
            neighborhood_model, neighborhood_dataloader.test_dataloader()
        ),
        dim=0,
    )
    truth = test_workload.graphlet_count_truth

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

    count_pred = 2 ** F.relu(count_pred) - 1

    count_pred = count_pred.cpu().numpy()
    truth = truth.cpu().numpy()
    norm_mse_LRP = norm_mse(pred=count_pred, truth=truth, groupby=groupby_list)
    print("norm_mse:", norm_mse_LRP)
    mae_LRP = mae(pred=count_pred, truth=truth, groupby=groupby_list)
    print("mae:", mae_LRP)

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

    # set args for ablation study model
    args_neighborhood.use_hetero = True
    args_neighborhood.use_tconv = True
    args_opt.test_gossip = False
    args_opt.train_gossip = False
    args_neighborhood.conv_type = "SAGE"
    args_neighborhood.use_canonical = False

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
