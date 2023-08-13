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

from subgraph_counting.config import (
    parse_encoder,
    parse_count,
    parse_optimizer_baseline,
)
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (
    LRPModel,
    DIAMNETModel,
)
from subgraph_counting.lightning_data import (
    LightningDataLoader,
    LightningDataLoader_LRP,
)
from subgraph_counting.transforms import get_truth
from subgraph_counting.workload import Workload_baseline, graph_atlas_plus
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.analysis import norm_mse, mae


class DIAMNet_args:
    def __init__(self) -> None:
        self.hidden_dim = 128
        self.dropout = 0.0
        self.layer_num = 5
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
    LRP: bool = False,
    DIAMNET: bool = True,
    train: bool = True,
    test: bool = True,
    train_dataset_name: str = None,
    test_dataset_name: str = None,
    checkpoint=None,
    nx_queries: List[nx.Graph] = None,
    atlas_query_ids: List[int] = None,
    output_dir: str = "results/raw",
):
    """
    train and test baselines
    """
    parser = argparse.ArgumentParser(description="Order embedding arguments")
    parse_optimizer_baseline(parser)
    parse_encoder(parser)

    model_args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Graphlet Count")
    parse_optimizer_baseline(parser)
    parse_count(parser)
    args = parser.parse_args()

    if LRP:
        baseline_args = LRP_args()
    elif DIAMNET:
        baseline_args = DIAMNet_args()
    for key in baseline_args.__dict__:
        setattr(model_args, key, getattr(baseline_args, key))

    print("model_args:", model_args)
    print("args:", args)

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

    if not LRP and not DIAMNET:
        raise ValueError("LRP and DIAMNET cannot be both False")
    if LRP and DIAMNET:
        raise ValueError("LRP and DIAMNET cannot be both True")

    # select gpu number
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    if train:
        if LRP:
            model = LRPModel(1, baseline_args.hidden_dim, model_args)
            # model.load_state_dict(
            #     torch.load("ckpt/debug/LRP_345_synXL_qs_epo50_init.pt")["state_dict"]
            # )
        elif DIAMNET:
            model = DIAMNETModel(
                1, baseline_args.hidden_dim, model_args, baseline="DIAMNet"
            )

    else:
        if LRP:
            model = LRPModel.load_from_checkpoint(checkpoint)
        elif DIAMNET:
            model = DIAMNETModel.load_from_checkpoint(checkpoint)
    model.set_queries(query_ids, device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)

    val_dataset_name = train_dataset_name

    if train:
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
        if LRP:
            train_workload.generate_LRP_dataset(query_ids=query_ids, args=args)
        elif DIAMNET:
            train_workload.generate_DIAMNET_dataset(query_ids=query_ids, args=args)

        val_dataset = train_dataset

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
        if LRP:
            val_workload.generate_LRP_dataset(query_ids=query_ids, args=args)
        elif DIAMNET:
            val_workload.generate_DIAMNET_dataset(query_ids=query_ids, args=args)

    if test:
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
        if LRP:
            test_workload.generate_LRP_dataset(query_ids=query_ids, args=args)
        elif DIAMNET:
            test_workload.generate_DIAMNET_dataset(query_ids=query_ids, args=args)

    if LRP:
        dataloader = LightningDataLoader_LRP(
            train_dataset=train_workload.LRP_dataset if (train) else None,
            val_dataset=val_workload.LRP_dataset if (train) else None,
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
            train_dataset=train_workload.DIAMNET_dataset if (train) else None,
            val_dataset=val_workload.DIAMNET_dataset if (train) else None,
            test_dataset=test_workload.DIAMNET_dataset,
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

    args.model_path = "ckpt/general/baseline/test"
    device_num = args.gpu.split(":")[-1]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[int(device_num)],
        max_epochs=args.num_epoch,
        callbacks=[checkpoint_callback],
        default_root_dir=args.model_path,
        log_every_n_steps=5,
    )

    if train:
        trainer.fit(model, dataloader)
    # if test:
    #     trainer.test(model, dataloader)

    # test dataset name
    print("test dataset name: ", test_dataset_name)
    # get predict count
    count_pred = torch.cat(trainer.predict(model, dataloader.test_dataloader()), dim=0)
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
        count_pred = 2 ** F.relu(count_pred) - 1
        truth = 2**truth - 1

    count_pred = count_pred.cpu().numpy()
    truth = truth.cpu().numpy()

    norm_mse_LRP = norm_mse(pred=count_pred, truth=truth, groupby=groupby_list)
    print("norm_mse:", norm_mse_LRP)
    mae_LRP = mae(pred=count_pred, truth=truth, groupby=groupby_list)
    print("mae:", mae_LRP)

    # print("LRP_count_test shape", LRP_count_test.shape)
    # LRP_count_test = torch.cat(LRP_count_test, dim=0).cpu().numpy()


if __name__ == "__main__":
    # torch.random.manual_seed(0)
    # np.random.seed(0)
    # lightning random seed
    pl.seed_everything(0)

    atlas_graph = defaultdict(list)
    for i in range(4, 53):
        # for i in range(4,53):
        g = graph_atlas_plus(i)  # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_graph[len(g)].append(i)
    query_ids = atlas_graph[3] + atlas_graph[4] + atlas_graph[5]

    main(
        LRP=False,
        DIAMNET=True,
        train=True,
        test=True,
        train_dataset_name="Syn_1827",
        test_dataset_name="MUTAG",
        atlas_query_ids=query_ids,
        checkpoint=None,
    )

    # diamnet model: ckpt/general/baseline/DIAMNet/GIN_DIAMNet_345_syn_1827/lightning_logs/version_5/checkpoints/epoch=298-step=8671.ckpt
    # LRP model: ckpt/general/baseline/LRP/LRP_345_Syn_1827/lightning_logs/version_7/checkpoints/epoch=49-step=17600.ckpt
