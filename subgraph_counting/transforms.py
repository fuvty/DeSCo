from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import networkx as nx
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType, NodeType, QueryType
import orca


class ZeroNodeFeat(BaseTransform):
    """
    set the node feature to zero
    """

    def __init__(self, node_feat_name: str = "x"):
        self.node_feat_name = node_feat_name

    def __call__(self, data: Data):
        # assert type(data) == Union[Data, Batch]

        x = getattr(data, self.node_feat_name)
        setattr(data, self.node_feat_name, torch.zeros_like(x))

        return data


class ToTCONV(BaseTransform):
    """
    ALERT: only convert one types of node, use ToTCONVHetero to convert multiple types of node
    recommand call \'pyg_graph = T.ToUndirected()(pyg_graph)\' first
    a --> b
    |    /
    V   /
    c <-
    """

    def __init__(self, node_type: str = "count", node_attr: str = "x"):
        self.node_type = node_type
        self.node_attr = node_attr

    def __call__(self, data: HeteroData):
        edge_types = []

        for edge_type in data.metadata()[1]:
            if edge_type[0] == self.node_type and edge_type[2] == self.node_type:
                edge_types.append(edge_type)

        for edge_type in edge_types:
            edge_tensor = data[edge_type].edge_index

            src_type = edge_type[0]
            rel_type = edge_type[1]
            dst_type = edge_type[2]

            if edge_tensor.numel() == 0:
                # empty tensor
                del data[src_type, rel_type, dst_type]
                data[
                    src_type, rel_type + "_triangle", dst_type
                ].edge_index = edge_tensor
                data[src_type, rel_type + "_tride", dst_type].edge_index = edge_tensor
                continue

            edge_tensor, _ = pyg_utils.remove_self_loops(edge_tensor)

            adj_size = getattr(data[self.node_type], self.node_attr).shape[0]

            A = torch.sparse_coo_tensor(
                edge_tensor,
                torch.ones([edge_tensor.shape[-1]], device=edge_tensor.device),
                (adj_size, adj_size),
                device=edge_tensor.device,
                requires_grad=False,
            )
            A2 = torch.sparse.mm(A, A)
            Tri_And_Edge = (A * A2 + A).coalesce()  # Triangle = A*A2
            # edge_weight = Triangle[edge_index] # can't do it :(

            edge_list = []
            tri_edge_index = []
            for edge, value in zip(Tri_And_Edge.indices().T, Tri_And_Edge.values()):
                edge_list.append(edge)
                tri_edge_index.append(value)
            # assert (edge_tensor == torch.stack(edge_list, dim=0).T).all().item()
            edge_tensor = torch.stack(edge_list, dim=0).T
            tri_edge_index = torch.stack(tri_edge_index, dim=0) > 1  # (#E,1)

            # assign to original pyg graph
            del data[src_type, rel_type, dst_type]

            data[src_type, rel_type + "_triangle", dst_type].edge_index = edge_tensor[
                :, tri_edge_index
            ]  # size = 2*#triangle_edge
            data[src_type, rel_type + "_tride", dst_type].edge_index = edge_tensor[
                :, ~tri_edge_index
            ]  # size = 2*#tride_edge
        return data


class ToQconvHetero(BaseTransform):
    """
    convert pyg graph to qconv graph
    """

    def __init__(self, node_attr: str = "x"):
        self.node_attr = node_attr

    def __call__(self, data: HeteroData):
        node_types = data.metadata()[0]
        edge_types = data.metadata()[1]

        homoData = data.to_homogeneous(edge_attrs=None, node_attrs=None)
        nx_graph = pyg_utils.to_networkx(
            homoData,
            node_attrs=["node_type", "node_feature"],
            to_undirected=True,
            remove_self_loops=True,
        )
        counts = orca.orbit_counts("edge", 4, nx_graph)
        priority = [11, 10, 5, 6, 7, 9, 4, 8, 3, 2]
        cnt = 0
        for edge in nx_graph.edges:
            find = False
            for p in priority:
                if counts[cnt][p] > 0:
                    nx_graph.edges[edge]["node_type"] = "union_" + str(p)
                    find = True
                    break
            if not find:
                nx_graph.edges[edge]["node_type"] = "union_1"

            cnt += 1
        for node in nx_graph.nodes:
            nx_graph.nodes[node]["node_feature"] = torch.tensor(
                nx_graph.nodes[node]["node_feature"], dtype=torch.float32
            )
            nx_graph.nodes[node]["node_type"] = node_types[
                nx_graph.nodes[node]["node_type"]
            ]

        heteroData = NetworkxToHetero(
            nx_graph, type_key="node_type", feat_key="node_feature"
        )

        return heteroData


class ToTconvHetero(BaseTransform):
    """
    recommand call \'pyg_graph = T.ToUndirected()(pyg_graph)\' first
    a --> b
    |    /
    V   /
    c <-
    """

    def __init__(self, node_attr: str = "x"):
        self.node_attr = node_attr

    def __call__(self, data: HeteroData):
        node_types = data.metadata()[0]
        edge_types = data.metadata()[1]

        # edge_tensor, node_slices, edge_slices = to_homogeneous_edge_index(data.clone()) # avoid affecting the original data
        edge_tensor, node_slices, edge_slices = to_homogeneous_edge_index(
            data
        )  # ALERT: wil affect the original data

        # sort edge_tensor and get the indices
        sorted_edge_tensor, edge_tensor_sort_indices = pyg_utils.sort_edge_index(
            edge_tensor,
            edge_attr=torch.tensor(
                [i for i in range(edge_tensor.shape[-1])], device=edge_tensor.device
            ),
        )

        adj_size = max(node_slices.values(), key=lambda x: x[1])[
            1
        ]  # maximum node indicies

        A = torch.sparse_coo_tensor(
            edge_tensor,
            torch.ones([edge_tensor.shape[-1]], device=edge_tensor.device),
            (adj_size, adj_size),
            device=edge_tensor.device,
            requires_grad=False,
        )
        A2 = torch.sparse.mm(A, A)
        Tri_And_Edge = (
            A * A2 + A
        ).coalesce()  # Triangle = A*A2, Tri_And_Edge is sorted for sure
        # edge_weight = Triangle[edge_index] # can't do it :(

        edge_list = []
        tri_edge_index = []
        for edge, value in zip(Tri_And_Edge.indices().T, Tri_And_Edge.values()):
            edge_list.append(edge)
            tri_edge_index.append(value)
        # assert (sorted_edge_tensor == torch.stack(edge_list, dim=0).T).all().item()
        # edge_tensor = torch.stack(edge_list, dim=0).T
        tri_edge_index = torch.stack(tri_edge_index, dim=0) > 1  # (#E,1)
        # sort tri_edge_index to match edge_tensor
        tri_edge_index = tri_edge_index.gather(
            dim=-1, index=edge_tensor_sort_indices.argsort()
        )

        for edge_type, slices in edge_slices.items():
            src_type = edge_type[0]
            rel_type = edge_type[1]
            dst_type = edge_type[2]

            tri_edge_index_this_type = tri_edge_index[slices[0] : slices[1]]
            edge_tensor_this_type = edge_tensor[:, slices[0] : slices[1]]
            edge_tensor_this_type[0, :] = (
                edge_tensor_this_type[0, :] - node_slices[src_type][0]
            )  # map to node indicies of the corresponding type
            edge_tensor_this_type[1, :] = (
                edge_tensor_this_type[1, :] - node_slices[dst_type][0]
            )

            data[
                src_type, rel_type + "_triangle", dst_type
            ].edge_index = edge_tensor_this_type[
                :, tri_edge_index_this_type
            ]  # size = 2*#triangle_edge
            data[
                src_type, rel_type + "_tride", dst_type
            ].edge_index = edge_tensor_this_type[
                :, ~tri_edge_index_this_type
            ]  # size = 2*#tride_edge

            # assign to original pyg graph
            del data[src_type, rel_type, dst_type]

        return data


def to_homogeneous_edge_index(
    data: HeteroData,
) -> Tuple[Optional[Tensor], Dict[NodeType, Any], Dict[EdgeType, Any]]:
    # Record slice information per node type:
    cumsum = 0
    node_slices: Dict[NodeType, Tuple[int, int]] = {}
    for node_type, store in data._node_store_dict.items():
        num_nodes = store.num_nodes
        node_slices[node_type] = (cumsum, cumsum + num_nodes)
        cumsum += num_nodes

    # Record edge indices and slice information per edge type:
    cumsum = 0
    edge_indices: List[Tensor] = []
    edge_slices: Dict[EdgeType, Tuple[int, int]] = {}
    for edge_type, store in data._edge_store_dict.items():
        src, _, dst = edge_type
        offset = [[node_slices[src][0]], [node_slices[dst][0]]]
        offset = torch.tensor(offset, device=store.edge_index.device)
        edge_indices.append(store.edge_index + offset)

        num_edges = store.num_edges
        edge_slices[edge_type] = (cumsum, cumsum + num_edges)
        cumsum += num_edges

    edge_index = None
    if len(edge_indices) == 1:  # Memory-efficient `torch.cat`:
        edge_index = edge_indices[0]
    elif len(edge_indices) > 0:
        edge_index = torch.cat(edge_indices, dim=-1)

    return edge_index, node_slices, edge_slices


def to_device(data, device):
    if isinstance(data, pyg.data.Data) or isinstance(data, pyg.data.HeteroData):
        # pyg data batch
        return data.to(device)
    elif isinstance(data, tuple):
        if len(data) == 4:
            # lrp data batch
            batch, pooling_matrix, sp_matrices, label = data
            return batch, pooling_matrix, sp_matrices, label.to(device), device
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_truth(data):
    """
    get ground truth count from data
    """
    if isinstance(data, pyg.data.Data) or isinstance(data, pyg.data.HeteroData):
        return data.y
    elif isinstance(data, tuple):
        return data[3].squeeze(dim=1)  # label
    else:
        raise NotImplementedError


def NetworkxToHetero(
    nx_graph: Union[nx.Graph, nx.DiGraph],
    type_key: str = "type",
    feat_key: str = "feat",
) -> pyg.data.HeteroData:
    """
    input: networkx graph
    node_types in networkx graph
    edge_types in networkx graph, organized as [(from_node, edge_type, to_node)]
    ALERT: only convery other torch.Tensor type node feat in graph
    """

    nx_graph = nx_graph.to_directed()
    hetero_graph = pyg.data.HeteroData()

    hetero_edge_dict = defaultdict(
        list
    )  # use (from_node_type, edge_type, to_node_type) to add edge_index
    hetero_node_dict = defaultdict(
        dict
    )  # use node_type to classify nodes, [node]: new_id

    # assign node_id based on node type
    for node in nx_graph.nodes:
        if type_key not in nx_graph.nodes[node]:  # if no node_type exist
            nx_graph.nodes[node][type_key] = "union_node"
        node_type = nx_graph.nodes[node][type_key]

        node_id = len(hetero_node_dict[node_type])
        hetero_node_dict[node_type][node] = node_id

    # assign edge based on edge type
    for edge in nx_graph.edges:
        n0, n1 = edge
        try:
            edge_type = nx_graph.edges[n0, n1][type_key]
        except KeyError:  # if no edge_type exist
            edge_type = "union"
        hetero_edge_type = (
            nx_graph.nodes[n0][type_key],
            edge_type,
            nx_graph.nodes[n1][type_key],
        )
        hetero_edge_dict[hetero_edge_type].append(
            (
                hetero_node_dict[nx_graph.nodes[n0][type_key]][n0],
                hetero_node_dict[nx_graph.nodes[n1][type_key]][n1],
            )
        )

    # add data to hetero graph
    for node_type, node_dict in hetero_node_dict.items():
        node_list = list(node_dict.items())
        node_list.sort(key=lambda x: x[1])
        node_feats_x = []
        node_attrs_pyg = defaultdict(list)

        node_attrs = list(
            next(iter(nx_graph.nodes(data=True)))[-1].keys()
        )  # get node attrs in a list
        node_attrs.remove(type_key)  # node type key
        if (
            feat_key not in node_attrs
        ):  # add feat Tensor([0]) if no feat exist for nodes
            for node in nx_graph.nodes:
                nx_graph.nodes[node][feat_key] = torch.zeros(1)
        else:
            node_attrs.remove(feat_key)  # input x key

        for node, i in node_list:
            if feat_key is not None:
                node_feat_x = nx_graph.nodes[node][feat_key]
                if type(node_feat_x) != torch.Tensor:
                    node_feat_x = torch.tensor(node_feat_x)
                node_feats_x.append(node_feat_x.view(-1))
            for attr in node_attrs:
                node_attr_pyg = nx_graph.nodes[node][attr]
                if type(node_attr_pyg) != torch.Tensor:
                    node_attr_pyg = torch.tensor(node_attr_pyg)
                node_attrs_pyg[attr].append(node_attr_pyg.view(-1))

        if feat_key is not None:
            hetero_graph[node_type].node_feature = torch.stack(node_feats_x, dim=0)
        for attr in node_attrs:
            setattr(
                hetero_graph[node_type], attr, torch.stack(node_attrs_pyg[attr], dim=0)
            )

    for edge_type, edge_list in hetero_edge_dict.items():
        hetero_graph[
            edge_type[0], edge_type[1], edge_type[2]
        ].edge_index = torch.tensor(edge_list).T

    return hetero_graph


class Relabel(pyg.transforms.BaseTransform):
    r"""Performs tensor device conversion, either for all attributes of the
    :obj:`~torch_geometric.data.Data` object or only the ones given by
    :obj:`attrs`.

    Args:
        device (torch.device): The destination device.
        attrs (List[str], optional): If given, will only perform tensor device
            conversion for the given attributes. (default: :obj:`None`)
        non_blocking (bool, optional): If set to :obj:`True` and tensor
            values are in pinned memory, the copy will be asynchronous with
            respect to the host. (default: :obj:`False`)
    """

    def __init__(
        self,
        device: Union[int, str],
        non_blocking: bool = False,
        mode: str = "decreasing_degree",
    ):
        self.device = device
        self.non_blocking = non_blocking
        self.mode = mode.replace("_", " ")

    def __call__(self, data: pyg.data.Data):
        graph_nx = pyg.utils.to_networkx(data, to_undirected=False)
        # graph_nx = relabel_graph_nx(graph_nx, mode= self.mode)
        return from_networkx_reorder(graph_nx, mode=self.mode)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.mode})"


class RemoveSelfLoops(pyg.transforms.BaseTransform):
    r"""Remove all the self loops from the graph

    Args:
        device (torch.device): The destination device.
        non_blocking (bool, optional): If set to :obj:`True` and tensor
            values are in pinned memory, the copy will be asynchronous with
            respect to the host. (default: :obj:`False`)
    """

    def __init__(
        self,
        device: Union[int, str],
        non_blocking: bool = False,
    ):
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, data: pyg.data.Data):
        data.edge_index, _ = pyg.utils.remove_self_loops(data.edge_index)
        data = T.RemoveIsolatedNodes()(data)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.mode})"
