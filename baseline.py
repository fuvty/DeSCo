import os
import sys

import argparse
import datetime
from typing import List, Optional, Tuple, Union
from collections import defaultdict
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

from subgraph_counting.config import parse_encoder, parse_count, parse_optimizer_LRP
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (
    GossipCountingModel,
    NeighborhoodCountingModel,
    LRPModel,
)
from subgraph_counting.lightning_data import (
    LightningDataLoader,
    LightningDataLoader_LRP,
)
from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat, get_truth
from subgraph_counting.workload import Workload, graph_atlas_plus
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.analysis import norm_mse, mae


class DIAMNet_args:
    def __init__(self) -> None:
        self.hidden_dim = 128
        self.dropout = 0.0
        self.n_layers = 5
        # self.conv_type = 'RGIN'
        self.conv_type = "GIN"
        self.use_hetero = False


class LRP_args:
    def __init__(self) -> None:
        self.hidden_dim = 8
        self.dropout = 0.0
        self.n_layers = 8
        self.use_hetero = False


def main(
    LRP: bool = True,
    DIAMNET: bool = False,
    train_LRP: bool = True,
    test_LRP: bool = True,
    train_DIAMNET: bool = False,
    test_DIAMNET: bool = False,
    LRP_checkpoint=None,
    nx_queries: List[nx.Graph] = None,
    atlas_query_ids: List[int] = None,
    output_dir: str = "results/raw",
):
    """
    train and test baselines
    """
    parser = argparse.ArgumentParser(description="Order embedding arguments")
    parse_optimizer_LRP(parser)
    parse_encoder(parser)

    model_args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Canonical Count")
    parse_optimizer_LRP(parser)
    parse_count(parser)
    args = parser.parse_args()

    lrp_args = LRP_args()
    for key in lrp_args.__dict__:
        setattr(model_args, key, getattr(lrp_args, key))

    # define queries by atlas ids or networkx graphs
    if nx_queries is None and atlas_query_ids is not None:
        nx_queries = [nx.graph_atlas(i) for i in atlas_query_ids]
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

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    if train_LRP:
        LRP_model = LRPModel(1, lrp_args.hidden_dim, model_args)
        LRP_model.load_state_dict(
            torch.load("ckpt/debug/LRP_345_synXL_qs_epo50.pt")["state_dict"]
        )
    else:
        LRP_model = LRPModel.load_from_checkpoint(LRP_checkpoint)
    LRP_model.set_queries(query_ids, device)

    # for name, param in LRP_model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)

    train_dataset_name = "ENZYMES"
    val_dataset_name = "ENZYMES"
    test_dataset_name = "ENZYMES"

    if train_LRP:
        train_dataset = load_data(train_dataset_name, transform=None)

        train_workload = Workload(
            train_dataset,
            "data/" + train_dataset_name,
            hetero_graph=False,
            node_feat_len=-1,
        )

        args.dataset = train_dataset_name
        args.dataset_name = train_dataset_name

        train_workload.get_LRP_workload(query_ids, args)

        val_dataset = train_dataset

        val_workload = Workload(
            val_dataset,
            "data/" + val_dataset_name,
            hetero_graph=False,
            node_feat_len=-1,
        )

        args.dataset = val_dataset_name
        args.dataset_name = val_dataset_name

        val_workload.get_LRP_workload(query_ids, args)

    if test_LRP:
        test_dataset = load_data(test_dataset_name, transform=None)

        test_workload = Workload(
            test_dataset,
            "data/" + test_dataset_name,
            hetero_graph=False,
            node_feat_len=-1,
        )

        args.dataset = test_dataset_name
        args.dataset_name = test_dataset_name

        test_workload.get_LRP_workload(query_ids, args)

    if LRP:
        dataloader = LightningDataLoader_LRP(
            train_dataset=train_workload.LRP_dataset if (train_LRP) else None,
            val_dataset=val_workload.LRP_dataset if (train_LRP) else None,
            test_dataset=test_workload.LRP_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_cpu,
            shuffle=False,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="LRP_counting_val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    elif DIAMNET:
        dataloader = LightningDataLoader(
            train_dataset=train_workload.neighs_pyg if (train_DIAMNET) else None,
            val_dataset=val_workload.neighs_pyg if (train_DIAMNET) else None,
            test_dataset=test_workload.neighs_pyg,
            batch_size=args.batch_size,
            num_workers=args.num_cpu,
            shuffle=False,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="DIAMNET_counting_val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    else:
        raise ValueError("LRP or DIAMNET must be True")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[2],
        max_epochs=args.num_epoch,
        callbacks=[checkpoint_callback],
        default_root_dir=args.model_path,
        log_every_n_steps=5,
    )

    trainer.fit(LRP_model, dataloader)
    trainer.test(LRP_model, dataloader)

    # get predict count
    LRP_count_pred = torch.cat(
        trainer.predict(LRP_model, dataloader.test_dataloader()), dim=0
    )
    truth = [[] for _ in range(len(query_ids))]
    for batch in dataloader.test_dataloader():
        for i, query_id in enumerate(query_ids):
            truth[i].append(get_truth(batch)[:, i].view(-1, 1))
    truth = torch.stack(
        [torch.cat(truth[i], dim=0).view(-1) for i in range(len(query_ids))], dim=1
    )

    query_size_dict = {i: len(g) for i, g in enumerate(nx_queries)}
    size_order_dict = {
        size: i for i, size in enumerate(sorted(set(query_size_dict.values())))
    }
    groupby_list = [[] for _ in range(len(size_order_dict))]
    for i in query_size_dict.keys():
        groupby_list[size_order_dict[query_size_dict[i]]].append(i)

    if args.use_log:
        LRP_count_pred = 2 ** F.relu(LRP_count_pred) - 1
        truth = 2**truth - 1

    LRP_count_pred = LRP_count_pred.cpu().numpy()
    truth = truth.cpu().numpy()

    norm_mse_LRP = norm_mse(pred=LRP_count_pred, truth=truth, groupby=groupby_list)
    print("norm_mse_LRP:", norm_mse_LRP)
    mae_LRP = mae(pred=LRP_count_pred, truth=truth, groupby=groupby_list)
    print("mae_LRP:", mae_LRP)

    # print("LRP_count_test shape", LRP_count_test.shape)
    # LRP_count_test = torch.cat(LRP_count_test, dim=0).cpu().numpy()


if __name__ == "__main__":
    atlas_graph = defaultdict(list)
    for i in range(4, 1253):
        # for i in range(4,53):
        g = graph_atlas_plus(i)  # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_graph[len(g)].append(i)
    query_ids = atlas_graph[3] + atlas_graph[4] + atlas_graph[5]

    main(atlas_query_ids=query_ids)
