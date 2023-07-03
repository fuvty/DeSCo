from collections import OrderedDict
import os
import sys

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import warnings
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
import torch.utils.data as torch_data

from torch_geometric.data import Batch

from subgraph_counting.gnn_model import BaseGNN, LRP_GraphEmbModule
from subgraph_counting.transforms import NetworkxToHetero, to_device, get_truth
from subgraph_counting.workload import graph_atlas_plus
from subgraph_counting.utils import add_node_feat_to_networkx
from subgraph_counting.LRP_dataset import (
    LRP_Dataset,
    collate_lrp_dgl_light_index_form_wrapper,
)
from subgraph_counting.DIAMNet import DIAMNet


def gen_queries(
    query_ids: List[int],
    queries=None,
    transform=None,
    node_feat_len: int = 1,
    hetero=True,
    device="cpu",
) -> Tuple[List[pyg.data.data.Data], List[nx.Graph]]:
    """
    generate queries according to the atlas ids
    return a list of PyG graphs and a list of networkx graphs
    """
    if queries is None:
        # TODO: deprecate the query_ids based query generation
        # convert nx_graph queries to pyg
        queries_nx = [graph_atlas_plus(query_id) for query_id in query_ids]
        if node_feat_len != 1:
            # exist node feat
            queries_nx_with_feat = []
            for query_nx in queries_nx:
                queries_nx_with_feat.extend(
                    add_node_feat_to_networkx(
                        query_nx,
                        [t for t in np.eye(node_feat_len).tolist()],
                        node_feat_key="feat",
                    )
                )
            queries_nx = queries_nx_with_feat
    else:
        queries_nx = queries
    if hetero:
        queries_pyg = [
            NetworkxToHetero(query, type_key="type", feat_key="feat")
            for query in queries_nx
        ]
    else:
        queries_pyg = [
            pyg.utils.from_networkx(query).to(device) for query in queries_nx
        ]
        for query_pyg in queries_pyg:
            query_pyg.node_feature = torch.zeros(
                (query_pyg.num_nodes, 1), device=device
            )

    # for query_pyg in queries_pyg:
    # query_pyg['union_node'].node_feature = torch.zeros((query_pyg['union_node'].num_nodes, 1))

    if transform is not None:
        queries_pyg = [transform(query_pyg) for query_pyg in queries_pyg]

    return queries_pyg, queries_nx


class NeighborhoodCountingModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        """
        init the model using the following args:
        layer_num: number of layers
        hidden_dim: the hidden dimension of the model
        conv_type: type of convolution
        use_hetero: whether to use heterogeneous convolution
        dropout: dropout rate; WARNING: dropout is not used in the model
        optional args:
        baseline: the baseline model to use, choose from ["DIAMNet"]
        """
        super(NeighborhoodCountingModel, self).__init__()

        self.emb_with_query = False  # noqa
        self.query_loader = None
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.kwargs = kwargs
        self.args = args

        # set every hyperparameters from args
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.save_hyperparameters()

        # define embed model
        self.emb_model = BaseGNN(
            input_dim, hidden_dim, hidden_dim, args, emb_channels=hidden_dim, **kwargs
        )
        # args.use_hetero = False # query graph are homogeneous for now
        self.emb_model_query = BaseGNN(
            input_dim, hidden_dim, hidden_dim, args, emb_channels=hidden_dim, **kwargs
        )

        self.count_model = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * args.hidden_dim, 1),
        )

    def training_step(self, batch: Batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("neighborhood_counting_train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch: Batch, batch_idx):
        loss = self.test_forward(batch, batch_idx)
        self.log(
            "neighborhood_counting_test_loss",
            loss,
            batch_size=batch.num_graphs,
            sync_dist=True,
        )

    def validation_step(self, batch: Batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log(
            "neighborhood_counting_val_loss",
            loss,
            batch_size=batch.num_graphs,
            sync_dist=True,
        )

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
        )  # add schedular

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "neighborhood_counting_val_loss",
        }

    # definition of the model that can also be used by pytorch
    def embed_to_count(self, embs: Tuple[torch.Tensor]):
        try:
            if self.kwargs["baseline"] == "DIAMNet":
                # DIAMNet
                emb_target, emb_query = embs
                emb_target, target_lens = emb_target
                emb_query, query_lens = emb_query
                count = self.count_model(emb_query, query_lens, emb_target, target_lens)
            elif self.kwargs["baseline"] == "LRP":
                # LRP concat, same as baseline count
                raise KeyError
            else:
                raise NotImplementedError
        except KeyError:
            # baseline concat
            embs = torch.cat(embs, dim=-1)  # concat query and target
            count = self.count_model(embs)
        return count

    def predict_step(self, batch: Batch, batch_idx) -> torch.Tensor:
        return self.graph_to_count(batch)

    def graph_to_count(self, batch) -> torch.Tensor:
        """
        use 2^(pred+1) as the predition to test the model, which is the groud truth canonical count
        use ReLu as the activation function to prevent the negative count
        """

        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch.to(self.device)))
        emb_queries = torch.cat(emb_queries, dim=0)
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        pred_results = []
        for i, query_emb in enumerate(emb_queries):
            emb = (
                emb_targets,
                query_emb.expand_as(emb_targets),
            )  # a batch of (target, query) paris with batch size the same as target
            results = self.embed_to_count(emb)
            pred_results.append(results)
        pred_results = torch.cat(pred_results, dim=-1)

        pred_results = 2**pred_results - 1
        return pred_results

    def graph_to_embed(self, batch) -> torch.Tensor:
        emb = self.emb_model(batch)
        return emb

    def train_forward(self, batch: Batch, batch_idx):
        """
        use log2(truth+1) as the truth to train the model
        """
        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch.to(self.device)))
        emb_queries = torch.cat(emb_queries, dim=0)
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        loss_accumulate = []
        for i, query_emb in enumerate(emb_queries):
            emb = (
                emb_targets,
                query_emb.expand_as(emb_targets),
            )  # a batch of (target, query) paris with batch size the same as target
            results = self.embed_to_count(emb)
            truth = batch.y[:, i].view(-1, 1)  # shape (batch_size,1)

            ######## different at train and test ########
            loss = self.criterion(results, torch.log2(truth + 1))
            #############################################

            loss_accumulate.append(loss)
        loss = torch.mean(torch.stack(loss_accumulate))
        return loss

    def test_forward(self, batch: Batch, batch_idx):
        """
        use 2^(pred+1) as the predition to test the model
        use ReLu as the activation function to prevent the negative count
        """
        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch.to(self.device)))
        emb_queries = torch.cat(emb_queries, dim=0)
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        loss_accumulate = []
        for i, query_emb in enumerate(emb_queries):
            emb = (
                emb_targets,
                query_emb.expand_as(emb_targets),
            )  # a batch of (target, query) paris with batch size the same as target
            results = self.embed_to_count(emb)
            truth = batch.y[:, i].view(-1, 1)  # shape (batch_size,1)

            ######## different at train and test ########
            loss = self.criterion(F.relu(2 ** (results - 1)), truth)
            #############################################

            loss_accumulate.append(loss)
        loss = torch.mean(torch.stack(loss_accumulate))
        return loss

    def criterion(self, count, truth):
        # regression
        loss_regression = F.smooth_l1_loss(count, truth)
        loss = loss_regression
        return loss

    def set_queries(
        self, query_ids, queries=None, transform=None, hetero=True, device="cpu"
    ):
        queries_pyg, queries_nx = gen_queries(
            query_ids,
            queries,
            transform=transform,
            node_feat_len=self.input_dim,
            hetero=hetero,
            device=device,
        )
        min_len_neighbor = max(nx.diameter(query) for query in queries_nx)
        if self.depth < min_len_neighbor:
            warnings.warn(
                "neighborhood diameter {:d} is too small for the queries, the minimum is {:d}".format(
                    self.depth, min_len_neighbor
                )
            )
        self.query_loader = DataLoader(queries_pyg, batch_size=64)

    def get_query_emb(self):
        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch))
        emb_queries = torch.cat(emb_queries, dim=0)
        return emb_queries

    def to_hetero_old(self, tconv_target=False, tconv_query=False):
        if tconv_target:
            self.emb_model.gnn_core = pyg.nn.to_hetero(
                self.emb_model.gnn_core,
                (
                    ["count", "canonical"],
                    [
                        ("count", "union_triangle", "count"),
                        ("count", "union_tride", "count"),
                        ("count", "union_triangle", "canonical"),
                        ("count", "union_tride", "canonical"),
                        ("canonical", "union_triangle", "count"),
                        ("canonical", "union_tride", "count"),
                    ],
                ),
                aggr="sum",
            )
        else:
            self.emb_model.gnn_core = pyg.nn.to_hetero(
                self.emb_model.gnn_core,
                (
                    ["count", "canonical"],
                    [
                        ("count", "union", "canonical"),
                        ("canonical", "union", "count"),
                        ("count", "union", "count"),
                    ],
                ),
                aggr="sum",
            )

        if tconv_query:
            self.emb_model_query.gnn_core = pyg.nn.to_hetero(
                self.emb_model_query.gnn_core,
                (
                    ["union_node"],
                    [
                        ("union_node", "union_triangle", "union_node"),
                        ("union_node", "union_tride", "union_node"),
                    ],
                ),
                aggr="sum",
            )
        else:
            self.emb_model_query.gnn_core = pyg.nn.to_hetero(
                self.emb_model_query.gnn_core,
                (["union_node"], [("union_node", "union", "union_node")]),
                aggr="sum",
            )

        return self

    def to_hetero(self, order: int = 3, SHMP_target=False, SHMP_query=False):
        if SHMP_target:
            if order == 3:
                self.emb_model.gnn_core = pyg.nn.to_hetero(
                    self.emb_model.gnn_core,
                    (
                        ["count", "canonical"],
                        [
                            ("count", "union_triangle", "count"),
                            ("count", "union_tride", "count"),
                            ("count", "union_triangle", "canonical"),
                            ("count", "union_tride", "canonical"),
                            ("canonical", "union_triangle", "count"),
                            ("canonical", "union_tride", "count"),
                        ],
                    ),
                    aggr="sum",
                )
            elif order == 4:
                src_dst_list = [
                    ("count", "canonical"),
                    ("count", "count"),
                    ("canonical", "count"),
                ]
                self.emb_model.gnn_core = pyg.nn.to_hetero(
                    self.emb_model.gnn_core,
                    (
                        ["count", "canonical"],
                        [
                            (node_type[0], "union_" + str(num), node_type[1])
                            for node_type in src_dst_list
                            for num in range(1, 12)
                        ],
                    ),
                    aggr="sum",
                )
        else:
            self.emb_model.gnn_core = pyg.nn.to_hetero(
                self.emb_model.gnn_core,
                (
                    ["count", "canonical"],
                    [
                        ("count", "union", "canonical"),
                        ("canonical", "union", "count"),
                        ("count", "union", "count"),
                    ],
                ),
                aggr="sum",
            )

        if SHMP_query:
            if order == 3:
                self.emb_model_query.gnn_core = pyg.nn.to_hetero(
                    self.emb_model_query.gnn_core,
                    (
                        ["union_node"],
                        [
                            ("union_node", "union_triangle", "union_node"),
                            ("union_node", "union_tride", "union_node"),
                        ],
                    ),
                    aggr="sum",
                )
            elif order == 4:
                self.emb_model_query.gnn_core = pyg.nn.to_hetero(
                    self.emb_model_query.gnn_core,
                    (
                        ["union_node"],
                        [
                            ("union_node", "union_" + str(num), "union_node")
                            for num in range(1, 12)
                        ],
                    ),
                    aggr="sum",
                )

        else:
            self.emb_model_query.gnn_core = pyg.nn.to_hetero(
                self.emb_model_query.gnn_core,
                (["union_node"], [("union_node", "union", "union_node")]),
                aggr="sum",
            )

        return self

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        convert the GNN model to heterogeneous model according to the checkpoint
        """
        use_hetero = checkpoint["hyper_parameters"]["args"].use_hetero
        use_tconv = checkpoint["hyper_parameters"]["args"].use_tconv
        self = (
            self.to_hetero_old(tconv_target=use_tconv, tconv_query=use_tconv)
            if use_hetero
            else self
        )
        return None


class GossipCountingModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        super(GossipCountingModel, self).__init__()
        self.hidden_dim = hidden_dim

        kwargs["baseline"] = "gossip"  # return all emb when forwarding

        # set every hyperparameters from args
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.save_hyperparameters()

        self.emb_model = BaseGNN(
            input_dim, hidden_dim, 1, args, **kwargs
        )  # output count
        self.kwargs = kwargs

    def training_step(self, batch: Batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("gossip_counting_train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch: Batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("gossip_counting_test_loss", loss, batch_size=batch.num_graphs)

    def validation_step(self, batch: Batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("gossip_counting_val_loss", loss, batch_size=batch.num_graphs)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
        )  # add schedular

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "gossip_counting_val_loss",
        }

    def train_forward(self, batch: Batch, batch_idx):
        """
        use the raw truth and pred count to train the model
        """
        loss_queries = []

        for query_id in range(batch.x.shape[1]):
            batch.node_feature = batch.x[:, query_id].view(-1, 1)
            query_emb = (
                self.query_emb[query_id, :].view(1, -1).detach().to(self.device)
            )  # with shape #query * feature_size, do not update query emb here; used by GossipConv

            neigh_pred = batch.x[:, query_id].view(-1, 1)
            gossip_pred = self.emb_model(batch, query_emb=query_emb)
            ground_truth = batch.y[:, query_id].view(-1, 1)

            loss = self.criterion(gossip_pred + neigh_pred, ground_truth)  # pred diff
            loss_queries.append(loss)

            # print(pred_counts.view(-1),torch.log2(batch.y[:,query_id]+1).view(-1),loss)

        loss = torch.sum(torch.stack(loss_queries))
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.graph_to_count(batch)

    def graph_to_count(self, batch, query_emb=None) -> torch.Tensor:
        pred_results = []
        for query_id in range(batch.x.shape[1]):
            batch.node_feature = batch.x[:, query_id].view(-1, 1)
            query_emb = (
                self.query_emb[query_id, :].view(1, -1).detach().to(self.device)
            )  # with shape #query * feature_size, do not update query emb here; used by GossipConv

            neigh_pred = batch.x[:, query_id].view(-1, 1)
            gossip_pred = self.emb_model(batch, query_emb=query_emb)

            pred_results.append(neigh_pred + gossip_pred)

        pred_results = torch.cat(pred_results, dim=-1)  # shape (#nodes, #queries)
        return pred_results

    def criterion(self, count, truth):
        # regression
        # loss = F.smooth_l1_loss(count, truth)
        loss = torch.log2(torch.abs(count - truth) + 1)
        # loss = torch.clip(loss, -0.5, 0.5)
        return loss

    def set_query_emb(self, query_emb: torch.Tensor, query_ids=None, queries=None):
        self.query_emb = query_emb.detach()

    def _gate_value(self, query_emb) -> torch.Tensor:
        """
        get the gate for each query of each layer, return tensor with shape (#layers, #queries, 1)
        """
        assert self.conv_type == "GOSSIP"
        gates = []
        for layer in self.emb_model.gnn_core.convs:
            gates.append(layer._gate_value(query_emb))
        gates = torch.stack(gates, dim=0)
        return gates


class DIAMNETModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        super(DIAMNETModel, self).__init__()
        self.emb_with_query = False

        self.kwargs = kwargs
        self.args = args

        # set every hyperparameters from args
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.save_hyperparameters()

        self.emb_model = BaseGNN(
            input_dim, hidden_dim, hidden_dim, args, emb_channels=hidden_dim, **kwargs
        )
        self.emb_model_query = BaseGNN(
            input_dim, hidden_dim, hidden_dim, args, emb_channels=hidden_dim, **kwargs
        )

        self.count_model = DIAMNet(
            pattern_dim=hidden_dim,
            graph_dim=hidden_dim,
            hidden_dim=hidden_dim,
            recurrent_steps=3,
            num_heads=4,
            mem_len=4,
            mem_init="mean",
        )

    def training_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("DIAMNET_counting_train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("DIAMNET_counting_test_loss", loss)

    def validation_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("DIAMNET_counting_val_loss", loss)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=0.00001
        )  # add scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "DIAMNET_counting_val_loss",
        }

    def criterion(self, count, truth):

        # regression
        loss_regression = F.smooth_l1_loss(count, truth)

        loss = loss_regression
        # loss = loss_classification
        return loss

    def train_forward(self, batch, batch_idx):
        batch = to_device(batch, self.device)
        emb_queries = []
        query_lens = []
        for query_batch in self.query_loader:
            emb, query_len = self.emb_model_query(query_batch)
            emb_queries.append(emb)
            query_lens.append(query_len)
        emb_queries = torch.cat(emb_queries, dim=0)
        query_lens = torch.cat(query_lens, dim=0)
        emb_queries = [
            (emb_queries[i], query_lens[i]) for i in range(emb_queries.shape[0])
        ]  # List[Tuple[Tensor, Tensor]], each represent a graph
        emb_target = self.emb_model(batch)

        query_ids = self.query_ids
        query_embs = dict()
        for i, query_id in enumerate(query_ids):
            query_embs[query_id] = emb_queries[i]
        loss_queries = []
        for i, query_id in enumerate(query_ids):
            emb_query = (
                query_embs[query_id][0].repeat(emb_target[0].shape[0], 1, 1),
                query_embs[query_id][1].repeat(emb_target[1].shape[0], 1),
            )
            # emb = (emb_target, emb_query)
            # emb_target, emb_query = emb
            emb_target_cur, target_lens = emb_target
            emb_query_cur, query_lens = emb_query
            results = self.count_model(
                emb_query_cur, query_lens, emb_target_cur, target_lens
            )
            truth = get_truth(batch)[:, i].view(-1, 1)
            loss = self.criterion(results, truth)
            loss_queries.append(loss)
        loss = torch.mean(torch.stack(loss_queries))

        return loss

    def set_queries(self, query_ids: List[int], device: torch.device):
        queries_pyg = [
            pyg.utils.from_networkx(graph_atlas_plus(query_id)).to(device)
            for query_id in query_ids
        ]
        for query_pyg in queries_pyg:
            query_pyg.node_feature = torch.zeros(
                (query_pyg.num_nodes, 1), device=device
            )
        self.query_loader = DataLoader(queries_pyg, batch_size=64)
        self.query_ids = query_ids

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch = to_device(batch, self.device)
        emb_queries = []
        query_lens = []
        for query_batch in self.query_loader:
            emb, query_len = self.emb_model_query(query_batch)
            emb_queries.append(emb)
            query_lens.append(query_len)
        emb_queries = torch.cat(emb_queries, dim=0)
        query_lens = torch.cat(query_lens, dim=0)
        emb_queries = [
            (emb_queries[i], query_lens[i]) for i in range(emb_queries.shape[0])
        ]  # List[Tuple[Tensor, Tensor]], each represent a graph
        emb_target = self.emb_model(batch)

        query_ids = self.query_ids
        query_embs = dict()
        for i, query_id in enumerate(query_ids):
            query_embs[query_id] = emb_queries[i]
        pred_result = []
        for i, query_id in enumerate(query_ids):
            emb_query = (
                query_embs[query_id][0].repeat(emb_target[0].shape[0], 1, 1),
                query_embs[query_id][1].repeat(emb_target[1].shape[0], 1),
            )
            emb_target_cur, target_lens = emb_target
            emb_query_cur, query_lens = emb_query
            results = self.count_model(
                emb_query_cur, query_lens, emb_target_cur, target_lens
            )
            pred_result.append(results)
        pred_result = torch.cat(pred_result, dim=1)
        return pred_result


class LRPModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        super(LRPModel, self).__init__()

        self.emb_with_query = False

        self.kwargs = kwargs
        self.args = args

        # set every hyperparameters from args
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.save_hyperparameters()

        # define embed model
        self.emb_model = LRP_GraphEmbModule(
            lrp_in_dim=input_dim,
            hid_dim=hidden_dim,
            num_layers=args.n_layers,
            num_atom_type=input_dim,
            lrp_length=16,
            num_bond_type=1,
            num_tasks=1,
        )
        self.emb_model_query = LRP_GraphEmbModule(
            lrp_in_dim=input_dim,
            hid_dim=hidden_dim,
            num_layers=args.n_layers,
            num_atom_type=input_dim,
            lrp_length=16,
            num_bond_type=1,
            num_tasks=1,
        )

        self.count_model = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * hidden_dim, 1),
        )

    def load_state_dict(
        self, state_dict: OrderedDict[str, Tensor], strict: bool = True
    ):
        return super().load_state_dict(state_dict, strict)

    def training_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("LRP_counting_train_loss", loss, batch_size=batch[0].y.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("LRP_counting_test_loss", loss, batch_size=batch[0].y.shape[0])

    def validation_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log("LRP_counting_val_loss", loss, batch_size=batch[0].y.shape[0])

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
        )  # add schedular

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "LRP_counting_val_loss",
        }

    def criterion(self, count, truth):
        # regression
        loss_regression = F.smooth_l1_loss(count, truth)

        loss = loss_regression
        # loss = loss_classification
        return loss

    def train_forward(self, batch, batch_idx):

        batch = to_device(batch, self.device)
        emb_queries = []
        for query_batch in self.query_loader:
            query_batch = to_device(query_batch, self.device)
            emb_queries.append(self.emb_model_query(query_batch))
        emb_queries = torch.cat(emb_queries, dim=0)
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        loss_accumulate = []
        for i, query_emb in enumerate(emb_queries):
            emb = (
                emb_targets,
                query_emb.expand_as(emb_targets),
            )  # a batch of (target, query) paris with batch size the same as target
            emb = torch.cat(emb, dim=-1)
            results = self.count_model(emb)
            # truth = batch.y[:, i].view(-1, 1)  # shape (batch_size,1)
            truth = get_truth(batch)[:, i].view(-1, 1)

            loss = self.criterion(results, truth)

            loss_accumulate.append(loss)
        loss = torch.mean(torch.stack(loss_accumulate))
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        batch = to_device(batch, self.device)
        emb_queries = []
        for query_batch in self.query_loader:
            query_batch = to_device(query_batch, self.device)
            emb_queries.append(self.emb_model_query(query_batch))
        emb_queries = torch.cat(emb_queries, dim=0)
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        pred_results = []
        for i, query_emb in enumerate(emb_queries):
            emb = (
                emb_targets,
                query_emb.expand_as(emb_targets),
            )  # a batch of (target, query) paris with batch size the same as target
            emb = torch.cat(emb, dim=-1)
            results = self.count_model(emb)
            pred_results.append(results)
        pred_results = torch.cat(pred_results, dim=1)
        return pred_results

    def set_queries(self, query_ids: List[int], device: torch.device):
        queries_pyg = [
            pyg.utils.from_networkx(graph_atlas_plus(query_id)).to(device)
            for query_id in query_ids
        ]
        for query_pyg in queries_pyg:
            query_pyg.node_feature = torch.zeros(
                (query_pyg.num_nodes, 1), device=device
            )
        queries_pyg = [g.to("cpu") for g in queries_pyg]
        query_LRP = LRP_Dataset(
            "queries_" + str(len(queries_pyg)),
            graphs=queries_pyg,
            labels=[torch.zeros(1) for _ in range(len(queries_pyg))],
            lrp_save_path="/home/nfs_data/weichiyue/2021Summer/tmp_folder",
            lrp_depth=1,
            subtensor_length=4,
            lrp_width=3,
        )
        self.query_loader = torch_data.DataLoader(
            query_LRP,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_lrp_dgl_light_index_form_wrapper(4),
        )
