import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.loader import DataLoader

from subgraph_counting.gnn_model import BaseGNN
from subgraph_counting.transforms import NetworkxToHetero
from subgraph_counting.workload import graph_atlas_plus

def gen_queries(query_ids, queries=None):
    # begin {query}, commonly used for all
    # convert nx_graph queries to pyg
    queries_pyg = [NetworkxToHetero(graph_atlas_plus(query_id), type_key= 'type', feat_key= 'feat') for query_id in query_ids]
    
    for query_pyg in queries_pyg:
        query_pyg['union_node'].node_feature = torch.zeros((query_pyg['union_node'].num_nodes, 1))
    # end {query}, commonly used for all
    return queries_pyg

class NeighborhoodCountingModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        '''
        init the model using the following args:
        n_layers: number of layers
        hidden_dim: the hidden dimension of the model
        conv_type: type of convolution
        use_hetero: whether to use heterogeneous convolution
        dropout: dropout rate; WARNING: dropout is not used in the model
        optional args:
        baseline: the baseline model to use, choose from ["DIAMNet"]
        '''
        super(NeighborhoodCountingModel, self).__init__()

        self.emb_with_query = False
        self.kwargs = kwargs
        self.args = args
        self.query_loader = None
        self.hidden_dim = hidden_dim 

        self.save_hyperparameters()

        # define embed model
        self.emb_model = BaseGNN(input_dim, hidden_dim, hidden_dim, args, emb_channels= hidden_dim, **kwargs)
        # args.use_hetero = False # query graph are homogeneous for now
        self.emb_model_query = BaseGNN(input_dim, hidden_dim, hidden_dim, args, emb_channels= hidden_dim, **kwargs)

        self.count_model = nn.Sequential(nn.Linear(2*hidden_dim, 4*args.hidden_dim), nn.LeakyReLU(), nn.Linear(4*args.hidden_dim, 1))
        
    
    def training_step(self, batch, batch_idx):
        return self.train_forward(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        loss = self.test_forward(batch, batch_idx)
        self.log('neighborhood_counting_test_loss', loss, batch_size=64)

    def validation_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log('neighborhood_counting_val_loss', loss, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return [optimizer], [lr_scheduler]

    # definition of the model that can also be used by pytorch
    def embed_to_count(self, embs: Tuple[torch.Tensor]):
        try:
            if self.kwargs['baseline'] == "DIAMNet":
                # DIAMNet
                emb_target, emb_query = embs
                emb_target, target_lens = emb_target
                emb_query, query_lens = emb_query
                count = self.count_model(emb_query, query_lens, emb_target, target_lens)
            elif self.kwargs['baseline'] == "LRP":
                # LRP concat, same as baseline count
                raise KeyError
            else:
                raise NotImplementedError
        except KeyError:
            # baseline concat
            embs = torch.cat(embs, dim=-1) # concat query and target
            count = self.count_model(embs)
        return count

    def graph_to_count(self, batch):
        '''
        use 2^(pred+1) as the predition to test the model
        use ReLu as the activation function to prevent the negative count
        '''

        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch))
        emb_queries = torch.cat(emb_queries, dim=0) 
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        pred_results = []
        for i, query_emb in enumerate(emb_queries):
            emb = (emb_targets, query_emb.expand_as(emb_targets)) # a batch of (target, query) paris with batch size the same as target
            results = self.embed_to_count(emb)
            pred_results.append(results)
        pred_results = torch.cat(pred_results, dim=-1)

        pred_results = F.relu(2**pred_results-1)
        return pred_results

    def train_forward(self, batch, batch_idx):
        '''
        use log2(truth+1) as the truth to train the model
        '''
        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch.to(self.device)))
        emb_queries = torch.cat(emb_queries, dim=0) 
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        loss_accumulate = []
        for i, query_emb in enumerate(emb_queries):
            emb = (emb_targets, query_emb.expand_as(emb_targets)) # a batch of (target, query) paris with batch size the same as target
            results = self.embed_to_count(emb)
            truth = batch.y[:,i].view(-1,1) # shape (batch_size,1)

            ######## different at train and test ########
            loss = self.criterion(results, torch.log2(truth+1))
            #############################################

            loss_accumulate.append(loss)     
        loss = torch.mean(torch.stack(loss_accumulate))
        return loss

    def test_forward(self, batch, batch_idx):
        '''
        use 2^(pred+1) as the predition to test the model
        use ReLu as the activation function to prevent the negative count
        '''
        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch.to(self.device)))
        emb_queries = torch.cat(emb_queries, dim=0) 
        emb_targets = self.emb_model(batch)

        # iterate over #emb_queries * #emb_targets to compute the count
        loss_accumulate = []
        for i, query_emb in enumerate(emb_queries):
            emb = (emb_targets, query_emb.expand_as(emb_targets)) # a batch of (target, query) paris with batch size the same as target
            results = self.embed_to_count(emb)
            truth = batch.y[:,i].view(-1,1) # shape (batch_size,1)

            ######## different at train and test ########
            loss = self.criterion(F.relu(2**(results-1)), truth)
            #############################################

            loss_accumulate.append(loss)     
        loss = torch.mean(torch.stack(loss_accumulate))
        return loss

    def criterion(self, count, truth):
        # regression
        loss_regression = F.smooth_l1_loss(count, truth)
        loss = loss_regression
        return loss

    def set_queries(self, query_ids, queries=None):
        self.query_loader = DataLoader(gen_queries(query_ids, queries), batch_size= 64)

    def get_query_emb(self):
        emb_queries = []
        for query_batch in self.query_loader:
            emb_queries.append(self.emb_model_query(query_batch))
        emb_queries = torch.cat(emb_queries, dim=0) 
        return emb_queries

    def to_hetero(self, tconv_target=False, tconv_query=False):
        if tconv_target:
            self.emb_model.gnn_core = pyg.nn.to_hetero(self.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union_triangle', 'count'), ('count', 'union_tride', 'count'), ('count', 'union_triangle', 'canonical'), ('count', 'union_tride', 'canonical'), ('canonical', 'union_triangle', 'count'), ('canonical', 'union_tride', 'count')] ), aggr='sum')
        else:
            self.emb_model.gnn_core = pyg.nn.to_hetero(self.emb_model.gnn_core, (['count', 'canonical'], [('count', 'union', 'canonical'), ('canonical', 'union', 'count'), ('count', 'union', 'count')] ), aggr='sum')
        
        if tconv_query:
            self.emb_model_query.gnn_core = pyg.nn.to_hetero(self.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union_triangle', 'union_node'), ('union_node', 'union_tride', 'union_node')] ), aggr='sum')
        else:
            self.emb_model_query.gnn_core = pyg.nn.to_hetero(self.emb_model_query.gnn_core, (['union_node'], [('union_node', 'union', 'union_node')] ), aggr='sum')


class GossipCountingModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        super(GossipCountingModel, self).__init__()
        self.hidden_dim = hidden_dim

        kwargs['baseline'] = "gossip" # return all emb when forwarding
        
        # debug
        self.emb_with_query = False

        self.emb_model = BaseGNN(input_dim, hidden_dim, 1, args, **kwargs) # output count
        self.kwargs = kwargs
        self.args = args

    def training_step(self, batch, batch_idx):
        return self.train_forward(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log('gossip_counting_test_loss', loss, batch_size=64)

    def validation_step(self, batch, batch_idx):
        loss = self.train_forward(batch, batch_idx)
        self.log('gossip_counting_val_loss', loss, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def train_forward(self, batch, batch_idx):
        '''
        use log2(truth+1) as the truth to train the model
        use the original value as loss
        '''
        loss_queries = []

        for query_id in range(batch.x.shape[1]):
            batch.node_feature = batch.x[:,query_id].view(-1,1)
            if self.emb_with_query:
                query_emb= self.query_emb[query_id, :].view(-1,1).detach().to(self.device) # with shape #query * feature_size, do not update query emb here
            else:
                query_emb= None
            pred_counts = self.emb_model(batch, query_emb= query_emb)
            pred_counts = pred_counts + torch.log2(batch.y[:,query_id]+1).view(-1,1)
            loss = self.criterion(2**(pred_counts-1), batch.y[:,query_id].view(-1,1)) # pred diff
            loss_queries.append(loss)
        
        loss = torch.sum(torch.stack(loss_queries))
        return loss

    def embed_to_count(self, data, query_emb= None) -> torch.Tensor:
        out_count = self.emb_model(data, query_emb= query_emb)
        return out_count

    def criterion(self, count, truth):
        # regression
        loss = F.smooth_l1_loss(count, truth)
        return loss

    def set_query_emb(self, query_emb: torch.Tensor, query_ids=None, queries=None):
        self.query_emb = query_emb.detach()


