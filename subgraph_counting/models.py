import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from typing import Tuple
from torch_geometric.nn.conv import SAGEConv

from common import feature_preprocess
from common.models import SkipLastGNN, BaseGNN
from subgraph_counting.baseline.GNNSubstructure.model_lrp import LRP_GraphEmbModule

from subgraph_counting.baseline.NeuralSubgraphCount.DIAMNet import DIAMNet

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, args, task=['counting', 'classification'], **kwargs):
        super(MultiTaskModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.task_num = len(task)
        self.task = task
        self.num_class = 10
        self.args = args

        # self.log_vars = nn.Parameter(torch.zeros((self.task_num)))
        if args.conv_type == "FCONV":
            self.emb_with_query = True
        else:
            self.emb_with_query = False

        if "use_log" in kwargs.keys():
            if kwargs['use_log']:
                self.use_log= True
            else:
                self.use_log= False
        else:
            self.use_log= False
        
        self.emb_model = BaseGNN(input_dim, hidden_dim, hidden_dim, args, emb_channels= hidden_dim, **kwargs)
        args.use_hetero = False # query graph are homogeneous for now
        self.emb_model_query = BaseGNN(input_dim, hidden_dim, hidden_dim, args, emb_channels= hidden_dim, **kwargs)


        self.multimodel = dict()
        if 'counting' in self.task:
            self.counting_model = nn.Sequential(nn.Linear(2*hidden_dim, 4*args.hidden_dim), nn.LeakyReLU(), nn.Linear(4*args.hidden_dim, 1))
        if 'classification' in self.task:
            self.classification_model = nn.Sequential(nn.Linear(2*hidden_dim, 4*args.hidden_dim), nn.LeakyReLU(), nn.Linear(4*args.hidden_dim, self.num_class))

    def multitask_model(self, task: str):
        if task == 'counting':
            return self.counting_model
        if task == 'classification':
            return self.classification_model
    
    def forward(self, embs: Tuple[torch.Tensor]):
        '''
        input: embs: tuple of tensor target,query emb
        '''
        emb_tensor = torch.cat(embs, dim=-1).to(embs[0].device)
        results = dict()
        for task_name in self.task:
            results[task_name] = self.multitask_model(task_name)(emb_tensor)
        return results

    def criterion(self, results, truth):
        '''
        results: dict[name] = tensor
        truth: tensor, count of pattern
        '''
        losses = []
        if 'counting' in self.task:
            losses.append(F.smooth_l1_loss(results['counting'].view(-1,1), truth))
        if "classification" in self.task:
            # classification
            label = truth.detach().clone().view(-1)
            if hasattr(self, 'use_log'):
                if self.use_log:
                    label = torch.floor(label) # for log usage, truth may not be an integer
                else:
                    label = torch.floor(torch.log(label+1))
            else:
                label = torch.floor(torch.log(label+1))
            label[label>8] = 9
            losses.append(F.cross_entropy(results['classification'], label.type(torch.LongTensor).to(label.device)))

        loss = torch.sum(torch.stack(losses)) # sum all losses
        return loss

class BaseLineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        '''
        init the model using the following args:
        hidden_dim: the hidden dimension of the model
        dropout: dropout rate
        n_layers: number of layers
        conv_type: type of convolution
        use_hetero: whether to use heterogeneous convolution
        optional args:
        baseline: the baseline model to use, choose from ["DIAMNet"]
        '''
        self.emb_with_query = False

        self.kwargs = kwargs
        self.args = args

        super(BaseLineModel, self).__init__()
        self.hidden_dim = hidden_dim 

        # define embed model
        try:
            if self.kwargs['baseline'] == "DIAMNet":
                # same as other model
                raise KeyError
            elif self.kwargs['baseline'] == "LRP":
                self.emb_model = LRP_GraphEmbModule(lrp_in_dim=input_dim, hid_dim=hidden_dim, num_layers=args.n_layers, num_atom_type=input_dim, lrp_length=16, num_bond_type=1, num_tasks=1)
                self.emb_model_query = LRP_GraphEmbModule(lrp_in_dim=input_dim, hid_dim=hidden_dim, num_layers=args.n_layers, num_atom_type=input_dim, lrp_length=16, num_bond_type=1, num_tasks=1)
            else:
                raise NotImplementedError
        except KeyError:
            self.emb_model = BaseGNN(input_dim, hidden_dim, hidden_dim, args, emb_channels= hidden_dim, **kwargs)
            # args.use_hetero = False # query graph are homogeneous for now
            self.emb_model_query = BaseGNN(input_dim, hidden_dim, hidden_dim, args, emb_channels= hidden_dim, **kwargs)

        try:
            if self.kwargs['baseline'] == "DIAMNet":
                print("USING DIAMNet BASELINE")
                self.count_model = DIAMNet(pattern_dim= hidden_dim, graph_dim= hidden_dim, hidden_dim= hidden_dim, recurrent_steps= 3, num_heads= 4, mem_len= 4, mem_init= 'mean') # use to debug DIAMNet
            elif self.kwargs['baseline'] == "LRP":
                # same as other model
                raise KeyError
        except KeyError:
            self.count_model = nn.Sequential(nn.Linear(2*hidden_dim, 4*args.hidden_dim), nn.LeakyReLU(), nn.Linear(4*args.hidden_dim, 1))
        

    '''
    def embed(self, data):
        return self.emb_model(data)
    '''
    
    def forward(self, embs: Tuple[torch.Tensor]):
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

    def criterion(self, count, truth):
        
        # regression
        loss_regression = F.smooth_l1_loss(count, truth)

        loss = loss_regression
        # loss = loss_classification
        return loss


class GossipModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, args, **kwargs):
        super(GossipModel, self).__init__()
        self.hidden_dim = hidden_dim

        if args.conv_type == "PFCONV":
            self.emb_with_query = True
        else:
            self.emb_with_query = False
        
        kwargs['baseline'] = "gossip" # return all emb when forwarding
        self.emb_model = BaseGNN(input_dim, hidden_dim, 1, args, **kwargs) # output count
        self.kwargs = kwargs
        self.args = args
    
    def gossip_count(self, data, query_emb= None) -> torch.Tensor:
        out_count = self.emb_model(data, query_emb= query_emb)
        return out_count

    def criterion(self, count, truth):
        # regression
        loss = F.smooth_l1_loss(count, truth)
        return loss

class MotifCountModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(MotifCountModel, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.emb_model_query = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)

        self.num_class = 10

        self.count_model = nn.Sequential(nn.Linear(2*hidden_dim, 4*args.hidden_dim), nn.LeakyReLU(), nn.Linear(4*args.hidden_dim, 1))

        self.class_model = nn.Sequential(nn.Linear(2*hidden_dim, 4*args.hidden_dim), nn.LeakyReLU(), nn.Linear(4*args.hidden_dim, self.num_class))

    def forward(self, data):
        embs = self.emb_model(data)
        count = self.count_model(embs)
        return count

    def criterion(self, count, truth):
        # return F.mse_loss(count, truth)
        # return F.huber_loss(count, truth)
        return F.smooth_l1_loss(count, truth)

class CanonicalCountModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(CanonicalCountModel, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.count_model = nn.Sequential(nn.Linear(hidden_dim, 4*args.hidden_dim), nn.ReLU(), nn.Linear(4*args.hidden_dim, 1))

    def forward(self, data):
        embs = self.emb_model(data)
        count = self.count_model(embs)
        return count

    def criterion(self, count, truth):
        return F.mse_loss(count, truth)

class Margin2Count(nn.Module):
    '''
    given a number called margin, return the count of pattern
    '''
    def __init__(self, input_dim, hidden_dim, args):
        super(Margin2Count, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(256, 2))