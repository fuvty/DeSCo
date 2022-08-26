"""Defines all graph embedding models"""
from typing import Callable, Union

import networkx as nx
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import warnings


class BaseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, **kwargs):
        '''
        init baseline GNN with the following args:
        dropout: dropout rate
        n_layers: number of layers
        conv_type: type of convolution
        use_hetero: whether to use heterogeneous convolution
        '''
        super(BaseGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers
        self.conv_type = args.conv_type
        self.use_hetero = args.use_hetero

        self.args = args
        self.kwargs = kwargs

        self.gnn_core = BaseGNNCore(input_dim, hidden_dim, output_dim, args, **kwargs)

        post_input_dim = self.gnn_core.post_input_dim

        self.anchor_mlp = nn.Sequential(
            nn.Linear(post_input_dim, post_input_dim),
            nn.LeakyReLU(0.1)
        )

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.conv_type = args.conv_type
        self.kwargs = kwargs

    def forward(self, data, query_emb= None):
        # gnn core forward
        if self.use_hetero:
            x, edge_index = data.node_feature_dict, data.edge_index_dict
        else:
            x, edge_index = data.node_feature, data.edge_index
        
        # pre_mp, gnn layers
        emb = self.gnn_core.forward(x, edge_index, query_emb= query_emb)

        # use special weights for anchored node when baseline is not gossip or no baseline is specificly assigned
        if self.use_hetero:
            try:
                emb['canonical'] = self.anchor_mlp(emb['canonical'])
            except KeyError: # if no canonical node exists
                pass
        else:
            try:
                if self.kwargs['baseline'] != "gossip":
                    emb[data.node_feature[:,0]==1, :] = self.anchor_mlp(emb[data.node_feature[:,0]==1, :])
            except:
                emb[data.node_feature[:,0]==1, :] = self.anchor_mlp(emb[data.node_feature[:,0]==1, :])

        # processed as to_homogeneous graph from now 
        # TODO: use other way to judge whether to use heterogeneous conversion
        if self.use_hetero:
            data = data.to_homogeneous() 
            emb = torch.cat([v for v in emb.values()],dim=0)
        
        # batch = data.batch.to(data.edge_index.device)
        try:
            if self.kwargs['baseline'] == "gossip":
                emb = self.post_mp(emb)
            else:
                raise NotImplementedError
        except:
            emb = pyg_nn.global_add_pool(emb, data.batch)
            emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class BaseGNNCore(nn.Module):
    '''
    The core of BaseGNN. It is seperated from BaseGNN for it to convert to heterogeneous gnn by to_hetero()
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, args, **kwargs):
        super(BaseGNNCore, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers
        self.conv_type = args.conv_type
        self.use_hetero = args.use_hetero
        # self.baseline = kwargs['baseline']
        self.args = args
        self.kwargs = kwargs

        pre_dim_out = hidden_dim
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, pre_dim_out))
        
        if 'input_pattern_emb' in kwargs.keys():
            pre_dim_out += kwargs['emb_channels'] # add query_emb

        conv_model = self.build_conv_model(args.conv_type, 1)
        
        self.convs = nn.ModuleList()
        self.updates = nn.ModuleList()

        if self.conv_type == "GIN":
            self.eps = nn.ModuleList()
        
        self.input_pattern_emb= False # init with false, if input_pattern_emb is in kwargs, then set to True

        for l in range(args.n_layers):
            hidden_input_dim = hidden_dim
            if l==0:
                try:
                    if 'input_pattern_emb' in kwargs.keys():
                        print("add query embed as input feature")
                        self.input_pattern_emb = True
                        hidden_input_dim = hidden_dim + kwargs['emb_channels']
                except:
                    print("require kwargs: emb_channels ")
            if args.conv_type == "PNACONV":
                assert(conv_model == pyg_nn.PNAConv)
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                warnings.warn("use deg info from {} with {} towers".format(kwargs['dataset'], kwargs['towers']))
                self.convs.append(conv_model(hidden_input_dim, hidden_dim, aggregators=aggregators, scalers= scalers, deg=kwargs['deg'], towers=kwargs['towers'], pre_layers=1, post_layers=1, divide_input=False))
            elif args.conv_type == "GOSSIP":
                # when using GossipConv, need to add query embed as input feature
                emb_channels = kwargs["emb_channels"]
                self.convs.append(conv_model(hidden_input_dim, hidden_dim, emb_channels= emb_channels))
            else:
                # common SHMP
                # conv layers
                self.convs.append(conv_model(hidden_input_dim, hidden_dim, aggr='add'))
                # update layers
                if self.conv_type == "SAGE":
                    self.updates.append(nn.Linear(2*hidden_dim, hidden_dim))
                elif self.conv_type == "GIN":
                    self.updates.append(
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    )
                    self.eps.append(TrivalParam(0.0))
                elif self.conv_type == "GCN":
                    pass
                elif self.conv_type == 'GAT':
                    pass
                else:
                    raise NotImplementedError

        self.post_input_dim = hidden_dim * args.n_layers + pre_dim_out

        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.conv_type = args.conv_type
        self.kwargs = kwargs

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return lambda i, h, aggr: pyg_nn.GCNConv(i, h, normalize=False)
        elif model_type == "GIN":
            return lambda i, h, aggr: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)
                ))
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "GOSSIP":
            return GossipConv
        else:
            raise NotImplementedError
            print("unrecognized model type")

    def forward(self, x, edge_index, query_emb= None):
        x = self.pre_mp(x)

        try:
            if self.input_pattern_emb:
                assert(query_emb is not None)
                x = torch.cat((query_emb.expand(x.shape[0],-1), x), dim=-1).clone().detach()
        except AttributeError:
            # print("model has no attribute self.input_pattern_emb, re-generate model or simply ignore")
            pass
        # GossipConv Preprocess
        if self.conv_type == "GOSSIP":
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)
            edge_index = pyg_utils.to_undirected(edge_index)
            edge_weight = edge_index[0,:] < edge_index[1,:]
        # Regular Conv
        else:
            edge_weight = None

        emb = x

        # propogate forward gnn layers
        for i in range(len(self.convs)):
            if self.conv_type == "GOSSIP":
                x = self.convs[i](x, edge_index, edge_weight= edge_weight, query_emb= query_emb)
            else:
                x_neigh = self.convs[i](x, edge_index)
                if self.conv_type == "SAGE":
                    x = self.updates[i](torch.cat((x_neigh, x), dim=1))
                elif self.conv_type == "GIN":
                    x = self.updates[i](x_neigh + (1 + self.eps[i]() * x))
                    # x = self.updates[i](x_neigh + (1 + 0.0) * x)
                elif self.conv_type in ["GCN", "GAT"]:
                    # not used by SHMP, this would be fine
                    x = x_neigh
                else:
                    raise NotImplementedError
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)

        return emb



class GossipConv(pyg_nn.MessagePassing):
    '''
    Convolution with learnable gates, used by gossip correction.
    '''
    def __init__(self, in_channels, out_channels, emb_channels, aggr="add", **kwargs):
        super(GossipConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_com = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels, out_channels)

        self.lin_gate = nn.Sequential(
            nn.Linear(emb_channels, out_channels),
            nn.Sigmoid(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid(),
            nn.LeakyReLU()
        )

    def forward(self, x, edge_index, edge_weight=None,
                size=None, res_n_id=None, query_emb=None):
        """
        Args:
            edge_weight = [edge_weight, query_emb]
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        
        # calculate information flow
        if edge_weight == None:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)
            edge_index = pyg_utils.to_undirected(edge_index)
            edge_weight = edge_index[0,:] < edge_index[1,:]
            warnings.warn("edge_weight should be computed before calling this function; it's a bool tensor of size #E, indicating the direction of each edge")

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id, query_emb=query_emb)

    def message(self, x_i, x_j, edge_weight, query_emb: Union[torch.Tensor, None]):
        if query_emb is None:
            gate = 0.5
            warnings.warn("lack query embed, use 0.5 for all queries.")
        else:
            gate = self.lin_gate(query_emb)
        edge_msg = self.lin_com(x_j)
        edge_msg[edge_weight] *= gate
        edge_msg[~edge_weight] *= (1-gate)
        return edge_msg

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)
        aggr_out = self.lin_update(aggr_out)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add", **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        # self.lin_update = nn.Linear(out_channels + in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        if isinstance(x, torch.Tensor):
            x = (x, x)
            
        edge_index = torch.zeros((2,0), dtype=torch.long, device=x[0].device) if edge_index is None else edge_index # fillin empty edge_index if it doesn't exist
        if torch.numel(edge_index) != 0:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        out = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id)
        out = self.lin(out)

        # out = torch.cat([out, x[1]], dim=-1)
        # return self.lin_update(out)

        return out

    def message(self, x_j, edge_weight):
        #return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return x_j

    '''
    def update(self, aggr_out, x, res_n_id):
        aggr_out = self.lin_update(aggr_out)
        return aggr_out
    '''
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        # self.lin_update.reset_parameters()

class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.initial_eps = eps
        # if train_eps:
        #     self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        # else:
        #     self.register_buffer('eps', torch.Tensor([eps]))
        
    def forward(self, x, edge_index, size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        '''
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        '''

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j



class TrivalParam(nn.Module):
    def __init__(self, n=0.0):
        super().__init__()
        self.n = n
        self.register_buffer('eps', torch.Tensor([n]))

    def forward(self) -> torch.Tensor:
        return self.eps

    def reset_parameters(self):
        self.eps.data.fill_(self.n)

