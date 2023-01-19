import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import concurrent.futures
import math
import multiprocessing as mp
import pickle
import random
import signal
import time
import warnings
from collections import defaultdict
from ctypes.wintypes import INT
from typing import List, Optional, Tuple, Union

import deepsnap as ds
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_scatter import segment_csr
from tqdm import tqdm
from tqdm.contrib import tzip

from subgraph_counting.data import (
    SymmetricFactor,
    count_canonical,
    count_graphlet,
    get_neigh_canonical,
    get_neigh_hetero,
    load_data,
    sample_graphlet,
    sample_neigh_canonical,
)
from subgraph_counting.transforms import NetworkxToHetero, Relabel, RemoveSelfLoops
from subgraph_counting.utils import add_node_feat_to_networkx


class GossipDataset(pyg.data.InMemoryDataset):
    """
    basically the same as pyg.data.Dataset.
    with addtional fuctions
    """

    def __init__(
        self,
        dataset,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        hetero_graph=True,
    ) -> None:
        self.dataset = dataset
        self.hetero_graph = hetero_graph
        # self.node_feat = node_feat
        super(GossipDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # suffix = '_node_feat' if self.node_feat else ''
        suffix = ""
        return ["gossip_pyg" + suffix + ".pt"]

    def process(self):
        """
        transform to gossip model
        """
        data_list = [g for g in self.dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return data

    def apply_truth_from_dataset(self, truth):
        try:
            self.slices["y"] = self.slices["x"]
        except:
            print(
                "Did not find attribute slices['x'] in gossip dataset. Please check if the dataset is processed correctly. If there is only one graph in the dataset, it is normal"
            )
        self.data.y = truth

    def apply_neighborhood_count(self, count: torch.Tensor, neighborhood_indicator):
        num_neighborhood, num_query = count.shape
        num_node = len(neighborhood_indicator)
        self.data.x = torch.zeros(num_node, num_query)
        self.data.x[neighborhood_indicator, :] = count.detach()

        self.slices["x"] = self.slices["y"]

    def aggregate_neighborhood_count(self, count: torch.Tensor) -> torch.Tensor:
        """
        aggregate count of neighborhoods to each graph, return tensor with shape (#graph, #query)
        """

        def expand_left(ptr: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
            for _ in range(dims + dim if dim < 0 else dim):
                ptr = ptr.unsqueeze(0)
            return ptr

        ptr = self.slices["y"]
        ptr = expand_left(ptr, 0, dims=count.dim())
        return segment_csr(count, ptr, reduce="sum")
        count_graphlet = scatter(count, ptr=self.slices["y"])
        return count_graphlet


class NeighborhoodDataset(pyg.data.InMemoryDataset):
    """
    get a normal pyg dataset and transform it into a canonical neighborhood dataset to feed to dataloader.
    """

    def __init__(
        self,
        depth_neigh,
        root,
        dataset: pyg.data.dataset.Dataset = None,
        nx_targets=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        hetero_graph=True,
        node_feat=False,
    ):
        self.nx_targets = nx_targets
        self.hetero_graph = hetero_graph
        self.node_feat = node_feat
        self.dataset = dataset
        self.depth_neigh = depth_neigh
        super(NeighborhoodDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.nx_neighs_index = np.load(
            self.processed_paths[1]
        )  # numpy 2D int array to show (graph_id, node_id) of each neighborhood, with shape (#neighborhood, 2)
        self.nx_neighs_indicator = np.load(
            self.processed_paths[2]
        )  # numpy bool array of shape (#n), indicating wheather the node is chosen as a neighborhood. size = #node in the dataset.

        self.node_feat_key = "feat"
        # TODO: always set node_feat_key in dataset as 'feat' for now

    @property
    def processed_file_names(self):
        suffix = "_node_feat" if self.node_feat else ""
        return [
            "neighs_pyg_depth_" + str(self.depth_neigh) + suffix + ".pt",
            "neighs_index_depth_" + str(self.depth_neigh) + suffix + ".npy",
            "neighs_indicator_depth_" + str(self.depth_neigh) + suffix + ".npy",
        ]

    def process(self):
        """
        iterate through original dataset to get the canonical neighborhoods
        """
        if self.dataset is None:
            raise AttributeError("must create Neighborhood dataset with a PyG dataset")

        if self.nx_targets is None:
            nx_targets = [
                pyg.utils.to_networkx(
                    g, to_undirected=True, node_attrs=["x"] if self.node_feat else None
                )
                if type(g) == pyg.data.Data
                else g
                for g in self.dataset
            ]
            # TODO: make it better. Now, force change the name of node feat to 'feat' and make it as tensor
            for graph in nx_targets:
                for n, data in graph.nodes(data=True):
                    data[self.node_feat_key] = torch.tensor(data.pop("x"))
            self.nx_targets = nx_targets
        else:
            nx_targets = self.nx_targets

        # can also use cpp
        # https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cpu/neighbor_sample_cpu.cpp

        if self.hetero_graph:
            get_neigh_func = get_neigh_hetero
        else:
            get_neigh_func = get_neigh_canonical

        nx_neighs_index = []
        nx_neighs_indicator = (
            []
        )  # when iterating through nx_targets, indicate wheather the node is chosen as a neighborhood. size = #node in the dataset
        nx_neighs = []

        # sample neighs
        for gid, graph in tqdm(enumerate(nx_targets), total=len(nx_targets)):
            for node in graph.nodes:
                target_neigh = get_neigh_func(nx_targets[gid], node, self.depth_neigh)
                if (
                    len(target_neigh.edges) == 0
                ):  # do not add this neigh to neighs list, all counts of patterns are 0
                    nx_neighs_indicator.append(False)
                else:  # add canonical neigh for canonical inference
                    nx_neighs_indicator.append(True)
                    nx_neighs_index.append((gid, node))
                    nx_neighs.append(target_neigh)

        # convert to pyg graph
        neighs_pyg = []
        for g in nx_neighs:
            if self.hetero_graph:
                g = NetworkxToHetero(g, type_key="type", feat_key="feat")
            else:
                g = pyg.utils.from_networkx(g)
                g.node_feature = g.node_feature.unsqueeze(dim=-1)
            g.y = torch.empty([1], dtype=torch.double).reshape(1, 1)
            neighs_pyg.append(g)

        # add missing edge_index type
        if self.hetero_graph:
            edge_types = set()
            for g in neighs_pyg:
                edge_types.update(g.metadata()[1])
            for g in neighs_pyg:
                for edge_type in edge_types:
                    if edge_type not in g.metadata()[1]:
                        g[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        if self.pre_filter is not None:
            neighs_pyg = [data for data in neighs_pyg if self.pre_filter(data)]

        if self.pre_transform is not None:
            neighs_pyg = [self.pre_transform(data) for data in neighs_pyg]

        data, slices = self.collate(neighs_pyg)
        torch.save((data, slices), self.processed_paths[0])

        np.save(self.processed_paths[1], np.array(nx_neighs_index, dtype=int))
        np.save(self.processed_paths[2], np.array(nx_neighs_indicator, dtype=bool))

    def apply_truth_from_dataset(self, truth):
        """
        truth: numeric tensor with shape (#node, #query)
        indicator: bool tensor with shape (#node), indicating weather the node is sampled for the neighborhood
        """
        self.data.y = truth[self.nx_neighs_indicator, :]

    def aggregate_neighborhood_count(self, count: torch.Tensor) -> torch.Tensor:
        """
        use self.neighs_indicator to aggregate count of neighborhoods to each graph
        input:
            canonical count: tensor with shape (#node, #query)
        output:
            graphlet count tensor with shape (#graph, #query)
        """
        count = count.clone().cpu()
        num_neighborhoods, num_queries = count.shape
        num_graphs = (
            max(self.nx_neighs_index[:, 0]) + 1
            if self.dataset is None
            else len(self.dataset)
        )
        graph_id = torch.tensor(self.nx_neighs_index[:, 0], dtype=torch.long).cpu()
        count_graphlet = torch.zeros(
            (num_graphs, num_queries), dtype=torch.float, device="cpu"
        )
        count_graphlet.index_add_(dim=0, index=graph_id, source=count)

        return count_graphlet


def MatchSubgraphWorker(task):
    """
    calculate the canonical count for a given query
    input:
    task_queue: (tid, target, qid, query, node_feat_key), node_feat_key is the key of node feature to be considered in graph isomorphism
    output_queue: (tid, qid, count_dict)
    """
    tid, target, qid, query, node_feat_key = task
    node_match = (
        (lambda x, y: x[node_feat_key] == y[node_feat_key])
        if (node_feat_key is not None)
        else None
    )
    GraphMatcher = nx.algorithms.isomorphism.GraphMatcher(
        target, query, node_match=node_match
    )
    SBM_iter = GraphMatcher.subgraph_isomorphisms_iter()
    count_dict = defaultdict(int)
    for vmap in SBM_iter:
        canonical_node = max(vmap.keys())
        count_dict[canonical_node] += 1
    return (tid, qid, tuple((k, v) for k, v in count_dict.items()))


class InMemoryDatasetLoader(pyg.data.InMemoryDataset):
    def __init__(self, InMemoryData, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = InMemoryData


class Workload:
    """
    generate the workload for subgraph counting problem, including the neighs and the ground truth
    """

    def __init__(
        self,
        dataset: pyg.data.dataset.Dataset,
        root: str,
        hetero_graph: bool = True,
        node_feat_len: int = -1,
        **kwargs,
    ) -> None:
        # whether to generate hetero graph
        # node feat: whether to add node feature as input

        self.dataset = dataset
        self.root = root

        self.hetero_graph = hetero_graph
        self.node_feat = node_feat_len != -1
        # the args used by this workload
        self.queries = []
        # list of int, indicating the graph_atlas id of the query
        self.query_ids = []

        # ground truth for neighborhoods
        # tensor with shape (#neighborhoods, #query)
        self.canonical_count_truth = torch.tensor([[]])
        # self.count_motif_valid = torch.tensor([[]])
        # list[pyg.data.Data]
        self.neighborhood_dataset = None
        self.gossip_dataset = None
        # self.neighs_valid = []
        # list[nx.Graph], holding the feature of every node on original graph
        self.graphs_pyg = []
        # self.graphs_valid = []

        # neighs_index[k]=(gid, nid) means the k-th result of neighs_train is the nid node from gid graph
        self.neighs_index = np.array([[]])
        # self.neighs_index_valid = []

        self.nx_targets = None

        # default settings
        self.order_by_degree = False
        self.max_file_name_len = 30

        if self.node_feat:
            self.node_feat_len = node_feat_len
            print("process dataset with node feature length ", self.node_feat_len)
            self.node_feat_key = "feat"
        else:
            self.node_feat_len = 1
            self.node_feat_key = None

    def generate_pipeline_datasets(
        self,
        depth_neigh,
        neighborhood_transform=None,
        gossip_transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        # sample neigh and generate neighborhood dataset
        self.neighborhood_dataset = NeighborhoodDataset(
            depth_neigh=depth_neigh,
            root=os.path.join(self.root, "NeighborhoodDataset"),
            dataset=self.dataset,
            nx_targets=self.nx_targets,
            transform=neighborhood_transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            hetero_graph=self.hetero_graph,
            node_feat=self.node_feat,
        )

        # generate gossip dataset
        self.gossip_dataset = GossipDataset(
            dataset=self.dataset,
            root=os.path.join(self.root, "GossipDataset"),
            transform=gossip_transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            hetero_graph=self.hetero_graph,
        )

        # if groudtruth is given, apply it to neighborhood dataset and gossip dataset
        if len(self.canonical_count_truth) != 0:
            self.neighborhood_dataset.apply_truth_from_dataset(
                self.canonical_count_truth
            )
            self.gossip_dataset.apply_truth_from_dataset(self.canonical_count_truth)

    def load_groundtruth(self, query_ids, queries=None):
        """
        load ground truth from file if exist;
        if file does not exist, return false
        """
        # TODO: allow load partial ground truth
        folder_path = os.path.join(self.root, "CanonicalCountTruth")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        suffix = "_node_feat" if self.node_feat else ""

        if query_ids is None and queries is not None:
            # use queries
            file_name = (
                "query_num_{:d}_".format(len(queries))
                + "query_len_sum_"
                + str(sum([len(g) for g in queries]))
                + suffix
                + ".pt"
            )
        elif query_ids is not None and queries is None:
            file_name = (
                "query_num_{:d}_".format(len(query_ids))
                + "atlas_ids_"
                + "_".join(map(str, query_ids[: self.max_file_name_len]))
                + suffix
                + ".pt"
            )
        elif queries is not None and query_ids is not None:
            raise ValueError("nx_queries and atlas_query_ids cannot be both empty")
        else:
            raise ValueError("nx_queries and atlas_query_ids cannot be both None")

        if os.path.exists(os.path.join(folder_path, file_name)):
            return torch.load(os.path.join(folder_path, file_name))
        else:
            raise NotImplementedError

    def exist_groundtruth(self, query_ids, queries=None):
        """
        check if ground truth exists
        """
        folder_path = os.path.join(self.root, "CanonicalCountTruth")
        suffix = "_node_feat" if self.node_feat else ""

        if query_ids is None and queries is not None:
            # use queries
            file_name = (
                "query_num_{:d}_".format(len(queries))
                + "query_len_sum_"
                + str(sum([len(g) for g in queries]))
                + suffix
                + ".pt"
            )
        elif query_ids is not None and queries is None:
            file_name = (
                "query_num_{:d}_".format(len(query_ids))
                + "atlas_ids_"
                + "_".join(map(str, query_ids[: self.max_file_name_len]))
                + suffix
                + ".pt"
            )
        elif queries is not None and query_ids is not None:
            raise ValueError("nx_queries and atlas_query_ids cannot be both empty")
        else:
            raise ValueError("nx_queries and atlas_query_ids cannot be both None")

        return os.path.exists(os.path.join(folder_path, file_name))

    def compute_groundtruth(
        self, query_ids=None, queries=None, num_workers=-1, save_to_file=True
    ):
        """
        compute the ground truth canonical count for the given query_ids or queries
        """
        if query_ids is None and queries is None:
            raise ValueError("query_ids or queries must be given")
        elif query_ids is not None and queries is not None:
            raise ValueError("query_ids and queries cannot be given at the same time")

        if query_ids is not None:
            # gen queries based on query_id
            # TODO: deprecate the query-id based ground truth computation
            queries = [graph_atlas_plus(i) for i in query_ids]
            if self.node_feat:
                new_queries = []
                for query in queries:
                    new_queries.extend(
                        add_node_feat_to_networkx(
                            query,
                            [t for t in np.eye(self.node_feat_len).tolist()],
                            self.node_feat_key,
                        )
                    )
                queries = new_queries
                query_ids = [-i for i in range(len(queries))]
                use_query_ids = False
            else:
                use_query_ids = True
        elif queries is not None:
            # gen query_ids based on queries
            query_ids = [-i for i in range(len(queries))]
            use_query_ids = False

        # convert dataset
        if self.nx_targets is None:
            self.nx_targets = [
                pyg.utils.to_networkx(
                    g, to_undirected=True, node_attrs=["x"] if self.node_feat else None
                )
                if type(g) == pyg.data.Data
                else g
                for g in self.dataset
            ]
        if self.node_feat:
            # TODO: make it better. Now, force change the name of node feat from 'x' to 'feat' and make it as tensor
            for graph in self.nx_targets:
                for n, data in graph.nodes(data=True):
                    data["feat"] = data.pop("x")

        nx_targets = [
            g.copy() for g in self.nx_targets
        ]  # make copy so that count_qid is not stored

        # init count value to zero
        for graph in nx_targets:
            for node in graph.nodes:
                for qid in query_ids:
                    graph.nodes[node]["count_" + str(qid)] = 0.0

        # NOTE: debug, set query_ids and queries to computed value for now. assuming no additional information of canonical count.
        self.query_ids = query_ids
        self.queries = queries

        # compute symmetry_factor
        symmetry_factors = dict()
        for qid, query in zip(query_ids, queries):
            symmetry_factors[qid] = SymmetricFactor(query, self.node_feat_key)

        # generate groundtruth tasks
        print("create tasks")
        tasks = []

        for tid, target in enumerate(nx_targets):
            for qid, query in zip(query_ids, queries):
                tasks.append(
                    (tid, target, qid, query, self.node_feat_key)
                )  # copy graphs to ensure no dependency

        start_time = time.time()
        get_results = 0

        # start workers
        if num_workers == -1:
            num_workers = os.cpu_count()
        print(
            "start workers: "
            + str(num_workers)
            + " workers, for "
            + str(len(tasks))
            + " tasks"
        )

        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            for tid, qid, count_dict in tqdm(
                executor.map(MatchSubgraphWorker, tasks), total=len(tasks)
            ):
                get_results += 1
                for node, count in count_dict:
                    nx_targets[tid].nodes[node]["count_" + str(qid)] = count
        end_time = time.time()
        print(
            "\ntime for counting with VF2:",
            end_time - start_time,
            ", get result:",
            get_results,
        )

        # assign results to a tensor with shape (#node, #query)

        # debug
        if not self.query_ids == query_ids:
            raise NotImplementedError

        # convert target graph's node feature to tensor
        if self.node_feat:
            for graph in nx_targets:
                for n, data in graph.nodes(data=True):
                    if type(data[self.node_feat_key]) != torch.Tensor:
                        data[self.node_feat_key] = torch.tensor(
                            data[self.node_feat_key]
                        )

        count_motif = []
        for graph in nx_targets:
            for node in graph.nodes:
                count_node = []
                for qid in self.query_ids:  # NOTE: use self.query_ids here
                    count = (
                        graph.nodes[node]["count_" + str(qid)] / symmetry_factors[qid]
                    )
                    count_node.append(count)
                count_motif.append(count_node)
        count_motif = torch.tensor(count_motif)
        # self.canonical_count_truth = count_motif

        self.nx_targets = nx_targets

        if save_to_file:
            folder_path = os.path.join(self.root, "CanonicalCountTruth")
            suffix = "_node_feat" if self.node_feat else ""
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if use_query_ids:
                file_name = (
                    "query_num_{:d}_".format(len(query_ids))
                    + "atlas_ids_"
                    + "_".join(map(str, query_ids[: self.max_file_name_len]))
                    + suffix
                    + ".pt"
                )
            else:
                file_name = (
                    "query_num_{:d}_".format(len(queries))
                    + "query_len_sum_"
                    + str(sum([len(g) for g in queries]))
                    + suffix
                    + ".pt"
                )
            torch.save(count_motif, os.path.join(folder_path, file_name))

        return count_motif

    def apply_neighborhood_count(self, count):
        self.gossip_dataset.apply_neighborhood_count(
            count, self.neighborhood_dataset.nx_neighs_indicator
        )

    def to_networkx(self):
        nx_targets = [
            pyg.utils.to_networkx(
                g, to_undirected=True, node_attrs=["x"] if self.node_feat else None
            )
            if type(g) == pyg.data.Data
            else g
            for g in self.dataset
        ]
        return nx_targets

    def save(self, root_folder: str = None):
        if root_folder == None:
            root_folder = "/tmp/" + self.name
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
        # save data
        with torch.no_grad():
            torch.save(
                self.canonical_count_truth, os.path.join(root_folder, "count_motif.pt")
            )
            torch.save(
                InMemoryDatasetLoader.collate(self.neighborhood_dataset),
                os.path.join(root_folder, "neighs_pyg.pt"),
            )
        with open(os.path.join(root_folder, "query.pk"), "wb") as f:
            pickle.dump(
                (self.queries, self.query_ids), f, protocol=pickle.HIGHEST_PROTOCOL
            )
        with open(os.path.join(root_folder, "graphs_nx.pk"), "wb") as f:
            pickle.dump(
                (self.graphs_nx, self.neighs_index), f, protocol=pickle.HIGHEST_PROTOCOL
            )

    def load(
        self,
        root_folder: str = None,
        load_list: list = ["count_motif", "neighs_pyg", "query", "graphs_nx"],
    ):
        if root_folder == None:
            root_folder = "/tmp/" + self.name
        if not os.path.exists(root_folder):
            print("root_folder %s not found" % (root_folder))
            raise ValueError

        # load data
        with torch.no_grad():
            if "neighs_pyg" in load_list:
                self.neighborhood_dataset = InMemoryDatasetLoader(
                    torch.load(os.path.join(root_folder, "neighs_pyg.pt"))
                )
            if "count_motif" in load_list:
                self.canonical_count_truth = torch.load(
                    os.path.join(root_folder, "count_motif.pt")
                )
        if "query" in load_list:
            with open(os.path.join(root_folder, "query.pk"), "rb") as f:
                self.queries, self.query_ids = pickle.load(f)
        if "graphs_nx" in load_list:
            with open(os.path.join(root_folder, "graphs_nx.pk"), "rb") as f:
                self.graphs_nx, self.neighs_index = pickle.load(f)


def gen_neighborhoods_sample(
    len_neighbor, args
) -> tuple[list[nx.Graph], list[nx.Graph]]:
    """
    return the canonical neighborhood for training and validation
    """
    neighs_train = []
    neighs_valid = []

    if args.relabel_mode is not None:
        transform = Relabel("cpu", mode=args.relabel_mode)
        dataset = load_data(args.dataset, args.n_neighborhoods, transform=transform)
    else:
        dataset = load_data(
            args.dataset, args.n_neighborhoods, transform=None
        )  # n_neighborhoods is only needed for syn

    if args.objective == "canonical":
        """
        sample neighborhoods and use canonical objective
        """
        sample_func = sample_neigh_canonical
    elif args.objective == "graphlet":
        """
        use the whole graph as input
        """
        sample_func = sample_graphlet
    else:
        print(args.objective)
        raise NotImplementedError

    print("sample train set")
    for b in tqdm(range(args.n_neighborhoods)):
        neighs_train.append(sample_func(dataset, len_neighbor))
    print("avg num of edges: ", np.mean([len(g.edges) for g in neighs_train]))
    print("sample validation set")
    for b in tqdm(range(args.val_size)):
        neighs_valid.append(sample_func(dataset, len_neighbor))

    return neighs_train, neighs_valid


def relabel_graph_nx(graph: nx.Graph, mode):
    return nx.convert_node_labels_to_integers(
        graph, first_label=0, ordering=mode, label_attribute=None
    )


def from_networkx_reorder(
    G,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
    mode="sorted",
):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G, ordering=mode)
    G = G.to_directed() if not nx.is_directed(G) else G

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = sorted(list(G.edges(keys=False)))
    else:
        edges = sorted(list(G.edges))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError("Not all nodes contain the same attributes")
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError("Not all edges contain the same attributes")
        for key, value in feat_dict.items():
            key = f"edge_{key}" if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in data.items():
        try:
            data[key] = torch.tensor(value)
        except ValueError:
            pass

    data["edge_index"] = edge_index.view(2, -1)
    data = pyg.data.Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f"edge_{key}" if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def graph_atlas_plus(atlas_id):
    """
    if atlas_id < 1253, return graph_atlas_plus(atlas_id);
    else return user defined pattern with 8~14 nodes, with id 8E3 ~ 14E3,
    use x000 and x001 for default setting
    """
    edgelist_plus_dict = {
        8000: [(0, 7), (7, 3), (7, 4), (1, 6), (6, 5), (2, 4), (3, 5)],
        8001: [(0, 5), (5, 1), (5, 6), (2, 3), (3, 7), (7, 4), (7, 6), (4, 6)],
        8002: [(0, 7), (7, 1), (7, 3), (2, 6), (6, 5), (3, 4), (4, 5)],
        8003: [(0, 7), (7, 3), (7, 4), (1, 5), (5, 3), (2, 6), (6, 4)],
        8004: [(0, 7), (7, 3), (7, 4), (7, 6), (1, 5), (5, 6), (2, 3), (4, 6)],
        8005: [(0, 7), (7, 4), (7, 5), (1, 3), (3, 2), (2, 6), (6, 4), (6, 5), (4, 5)],
        8006: [(0, 6), (6, 1), (6, 7), (2, 4), (4, 7), (3, 5), (5, 7)],
        8007: [(0, 7), (7, 3), (7, 4), (7, 6), (1, 5), (5, 6), (2, 6), (3, 4)],
        9000: [(0, 7), (7, 6), (1, 4), (4, 8), (2, 5), (5, 8), (3, 6), (3, 8)],
        9001: [
            (0, 4),
            (4, 3),
            (1, 5),
            (1, 8),
            (5, 6),
            (5, 8),
            (8, 6),
            (8, 7),
            (2, 3),
            (2, 7),
            (7, 6),
        ],
        9002: [(0, 8), (8, 5), (8, 7), (1, 7), (7, 4), (2, 6), (6, 5), (3, 4)],
        9003: [(0, 5), (5, 8), (1, 6), (6, 3), (2, 7), (7, 4), (3, 8), (8, 4)],
        9004: [(0, 7), (7, 3), (7, 6), (1, 8), (8, 3), (8, 6), (2, 4), (4, 5), (5, 6)],
        9005: [(0, 8), (8, 2), (8, 7), (1, 6), (6, 2), (3, 5), (3, 7), (5, 4), (7, 4)],
        9006: [(0, 7), (7, 4), (7, 8), (1, 6), (6, 2), (6, 8), (3, 5), (5, 8), (4, 8)],
        9007: [
            (0, 7),
            (7, 2),
            (7, 8),
            (1, 3),
            (3, 8),
            (2, 8),
            (8, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ],
        10000: [(0, 8), (8, 5), (8, 9), (1, 7), (7, 6), (2, 5), (3, 4), (4, 9), (9, 6)],
        10001: [(0, 9), (9, 1), (9, 5), (9, 8), (2, 5), (3, 6), (6, 8), (4, 7), (7, 8)],
        10002: [(0, 8), (8, 1), (8, 9), (2, 6), (6, 7), (3, 4), (4, 9), (9, 5), (5, 7)],
        10003: [(0, 8), (8, 4), (8, 7), (1, 7), (2, 5), (5, 9), (3, 6), (6, 9), (4, 9)],
        10004: [
            (0, 8),
            (8, 5),
            (8, 6),
            (1, 7),
            (7, 3),
            (7, 9),
            (2, 5),
            (3, 9),
            (9, 4),
            (9, 6),
            (4, 6),
        ],
        10005: [
            (0, 8),
            (8, 5),
            (8, 6),
            (1, 5),
            (2, 7),
            (2, 9),
            (7, 6),
            (7, 9),
            (9, 3),
            (9, 4),
            (3, 4),
        ],
        10006: [
            (0, 9),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 8),
            (1, 8),
            (8, 7),
            (2, 5),
            (3, 6),
            (4, 7),
        ],
        10007: [
            (0, 3),
            (3, 9),
            (1, 4),
            (4, 9),
            (2, 5),
            (5, 9),
            (9, 8),
            (6, 7),
            (6, 8),
            (7, 8),
        ],
        11000: [
            (0, 9),
            (9, 1),
            (9, 8),
            (2, 8),
            (8, 10),
            (3, 5),
            (5, 10),
            (4, 7),
            (7, 6),
            (10, 6),
        ],
        11001: [
            (0, 10),
            (10, 6),
            (10, 8),
            (1, 9),
            (9, 2),
            (9, 8),
            (3, 8),
            (4, 7),
            (7, 5),
            (7, 6),
            (5, 6),
        ],
        11002: [
            (0, 8),
            (8, 1),
            (8, 9),
            (2, 10),
            (10, 5),
            (10, 6),
            (3, 7),
            (7, 9),
            (4, 6),
            (5, 9),
        ],
        11003: [
            (0, 10),
            (10, 5),
            (10, 6),
            (10, 9),
            (1, 8),
            (8, 5),
            (2, 7),
            (7, 9),
            (3, 9),
            (4, 6),
        ],
        11004: [
            (0, 10),
            (10, 1),
            (10, 4),
            (10, 5),
            (10, 6),
            (2, 8),
            (8, 7),
            (3, 9),
            (9, 4),
            (9, 7),
            (5, 6),
        ],
        11005: [
            (0, 10),
            (10, 5),
            (10, 6),
            (10, 7),
            (1, 9),
            (9, 5),
            (9, 8),
            (2, 7),
            (7, 6),
            (3, 8),
            (8, 4),
        ],
        11006: [
            (0, 10),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 8),
            (10, 9),
            (2, 7),
            (7, 5),
            (3, 6),
            (6, 9),
            (4, 5),
            (4, 8),
            (8, 9),
        ],
        11007: [
            (0, 9),
            (9, 8),
            (9, 10),
            (1, 5),
            (5, 10),
            (2, 6),
            (6, 10),
            (3, 7),
            (7, 10),
            (4, 8),
            (4, 10),
        ],
        12000: [
            (0, 10),
            (10, 1),
            (10, 5),
            (2, 9),
            (9, 6),
            (9, 11),
            (3, 6),
            (4, 8),
            (8, 7),
            (5, 11),
            (11, 7),
        ],
        12001: [
            (0, 11),
            (11, 3),
            (11, 6),
            (11, 7),
            (1, 10),
            (10, 6),
            (10, 8),
            (2, 9),
            (9, 5),
            (9, 7),
            (3, 4),
            (4, 7),
            (5, 8),
            (8, 6),
        ],
        12002: [
            (0, 8),
            (8, 9),
            (1, 10),
            (10, 2),
            (10, 11),
            (3, 6),
            (6, 11),
            (4, 7),
            (7, 11),
            (5, 9),
            (5, 11),
        ],
        12003: [
            (0, 11),
            (11, 6),
            (11, 9),
            (1, 10),
            (10, 5),
            (10, 9),
            (2, 5),
            (3, 6),
            (4, 7),
            (7, 8),
            (8, 9),
        ],
        12004: [
            (0, 6),
            (6, 7),
            (6, 9),
            (1, 10),
            (10, 8),
            (10, 9),
            (10, 11),
            (2, 11),
            (11, 8),
            (11, 9),
            (3, 5),
            (5, 4),
            (4, 7),
            (7, 8),
            (9, 8),
        ],
        12005: [
            (0, 11),
            (11, 2),
            (11, 8),
            (1, 7),
            (7, 9),
            (2, 8),
            (8, 10),
            (3, 4),
            (3, 10),
            (4, 5),
            (10, 9),
            (5, 6),
            (6, 9),
        ],
        12006: [
            (0, 11),
            (11, 3),
            (11, 6),
            (11, 8),
            (1, 10),
            (10, 5),
            (10, 7),
            (10, 8),
            (2, 6),
            (2, 9),
            (6, 9),
            (9, 5),
            (9, 7),
            (3, 4),
            (4, 8),
            (8, 7),
            (5, 7),
        ],
        12007: [
            (0, 9),
            (9, 7),
            (9, 10),
            (9, 11),
            (1, 11),
            (11, 2),
            (11, 7),
            (11, 10),
            (3, 5),
            (5, 10),
            (4, 6),
            (4, 8),
            (6, 10),
            (8, 7),
            (8, 10),
        ],
        13000: [
            (0, 12),
            (12, 4),
            (12, 5),
            (12, 6),
            (1, 11),
            (11, 2),
            (11, 7),
            (3, 9),
            (9, 8),
            (4, 10),
            (10, 7),
            (10, 8),
            (5, 6),
        ],
        13001: [
            (0, 9),
            (9, 10),
            (9, 11),
            (1, 7),
            (7, 3),
            (2, 8),
            (8, 4),
            (3, 12),
            (12, 4),
            (12, 10),
            (12, 11),
            (5, 10),
            (5, 11),
            (10, 6),
            (11, 6),
        ],
        13002: [
            (0, 12),
            (12, 1),
            (12, 6),
            (12, 10),
            (2, 11),
            (11, 7),
            (11, 10),
            (3, 6),
            (4, 7),
            (5, 8),
            (8, 9),
            (9, 10),
        ],
        13003: [
            (0, 6),
            (6, 7),
            (1, 7),
            (7, 8),
            (2, 11),
            (11, 3),
            (11, 9),
            (4, 12),
            (12, 5),
            (12, 10),
            (8, 9),
            (8, 10),
            (9, 10),
        ],
        13004: [
            (0, 12),
            (12, 8),
            (12, 9),
            (12, 10),
            (1, 11),
            (11, 4),
            (11, 8),
            (2, 4),
            (3, 9),
            (9, 7),
            (5, 6),
            (5, 10),
            (6, 7),
            (10, 8),
        ],
        13005: [
            (0, 12),
            (12, 7),
            (12, 8),
            (12, 9),
            (1, 11),
            (11, 5),
            (11, 6),
            (2, 9),
            (9, 5),
            (3, 10),
            (10, 4),
            (10, 7),
            (6, 8),
            (8, 7),
        ],
        13006: [
            (0, 9),
            (9, 6),
            (9, 10),
            (1, 7),
            (1, 12),
            (7, 10),
            (7, 11),
            (12, 2),
            (12, 3),
            (12, 6),
            (12, 10),
            (2, 3),
            (4, 5),
            (4, 11),
            (5, 8),
            (11, 8),
            (11, 10),
            (8, 6),
        ],
        13007: [
            (0, 12),
            (12, 1),
            (12, 2),
            (12, 5),
            (12, 9),
            (3, 11),
            (11, 4),
            (11, 5),
            (11, 9),
            (6, 7),
            (6, 10),
            (7, 8),
            (7, 10),
            (10, 8),
            (10, 9),
            (8, 9),
        ],
        14000: [
            (0, 13),
            (13, 6),
            (13, 7),
            (13, 11),
            (1, 12),
            (12, 2),
            (12, 11),
            (3, 8),
            (8, 11),
            (4, 9),
            (9, 6),
            (5, 10),
            (10, 7),
        ],
        14001: [
            (0, 13),
            (13, 6),
            (13, 7),
            (13, 11),
            (13, 12),
            (1, 8),
            (8, 9),
            (2, 11),
            (11, 10),
            (3, 12),
            (12, 4),
            (12, 5),
            (6, 10),
            (7, 9),
        ],
        14002: [
            (0, 13),
            (13, 6),
            (13, 7),
            (13, 10),
            (13, 11),
            (1, 12),
            (12, 2),
            (12, 10),
            (3, 9),
            (9, 10),
            (4, 7),
            (5, 8),
            (8, 11),
            (6, 11),
        ],
        14003: [
            (0, 12),
            (12, 1),
            (12, 10),
            (2, 7),
            (7, 11),
            (3, 8),
            (8, 5),
            (4, 9),
            (9, 6),
            (5, 13),
            (13, 6),
            (13, 10),
            (13, 11),
            (11, 10),
        ],
        14004: [
            (0, 13),
            (13, 7),
            (13, 9),
            (13, 10),
            (1, 11),
            (11, 8),
            (11, 10),
            (2, 9),
            (9, 12),
            (3, 12),
            (12, 4),
            (12, 5),
            (6, 7),
            (8, 10),
        ],
    }

    if atlas_id < 1253:
        return nx.graph_atlas(atlas_id)
    else:
        graph = nx.Graph(directed=False)
        graph.add_edges_from(edgelist_plus_dict[atlas_id])
        return graph


class NeighborhoodDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        test_dataset=None,
        val_dataset=None,
        batch_size=32,
        shuffle=True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    # def transfer_batch_to_device(self, batch: pyg.data.Batch, device, dataloader_idx):
    #     if isinstance(batch, pyg.data.Batch):
    #         batch = batch.to(device)
    #     else:
    #         batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
    #     return batch


if __name__ == "__main__":
    dataset = load_data("ENZYMES")

    # define workload
    # TODO: add valid set mask support
    workload = Workload(
        dataset,
        "/home/nfs_data/futy/repos/prime/GNN_Mining/2021Summer/data/ENZYMES",
        hetero_graph=True,
    )

    # add this line when you need to compute ground truth
    workload.compute_groundtruth(query_ids=[6, 7])

    # generate pipeline dataset
    workload.generate_pipeline_datasets(depth_neigh=4)

    # begin canonical count
    neighborhood_dataloader = DataLoader(
        workload.neighborhood_dataset, batch_size=10, shuffle=True
    )

    # canonical count training
    # TODO: canonical training

    # canonical count inference
    # TODO: canonical inference
    neighborhood_count = torch.rand(
        len(workload.neighborhood_dataset), 2
    )  # size = (#neighborhood, #queries)

    # apply neighborhood count output to gossip dataset
    workload.apply_neighborhood_count(neighborhood_count)

    # gossip training
    gossip_dataloader = DataLoader(workload.gossip_dataset)
    # TODO: canonical training

    # gossip inference

    raise NotImplementedError

    class Args:
        def __init__(self):
            self.dataset = "ENZYMES"
            self.n_neighborhoods = 6400
            self.objective = "canonical"
            self.relabel_mode = None
            self.use_log = True
            self.use_norm = False

    atlas_graph = defaultdict(list)
    for i in range(4, 1253):
        # for i in range(4,53):
        g = graph_atlas_plus(i)  # range(0,1253)
        if sum(1 for _ in nx.connected_components(g)) == 1:
            atlas_graph[len(g)].append(i)
    query_ids = atlas_graph[3] + atlas_graph[4] + atlas_graph[5]
    # query_ids = [81, 103, 276, 320, 8006, 8007, 9006, 9007, 10006, 10007, 11006, 11007, 12006, 12007, 13006, 13007]

    print("number of queries:", len(query_ids))

    args = Args()

    workload = Workload("runtime_test_only", sample_neigh=False, hetero_graph=True)
    workload.gen_workload_general(query_ids=query_ids, args=args)

    workload.save(
        "subgraph_counting/workload/general/ENZYMES_gossip_n_query_"
        + str(len(query_ids))
        + "_all_hetero"
    )

    # print(torch.sum(torch.round(2**workload.count_motif-1), dim=0))
    # print(torch.sum(workload.count_motif, dim=0))

    # num_node = []
    # for neigh in workload.neighs_pyg:
    #     num_node.append(len(neigh['count'].node_feature)+len(neigh['canonical'].node_feature))
    # num_node.append(len(neigh.node_feature))

    # print(sum(num_node)/len(workload.neighs_pyg))
    # print(max(num_node))

    print("done")
