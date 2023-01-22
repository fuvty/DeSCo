from math import ceil
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import gzip
import multiprocessing as mp
import random
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph as DSGraph
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.datasets import Entities, Planetoid, TUDataset
from tqdm import tqdm

from subgraph_counting import combined_syn
from subgraph_counting.transforms import Relabel


def gen_query_ids(query_size: List[int]) -> List[int]:
    """
    input: query_size, a list of query size
    output: query_ids, a list of query ids
    """
    query_ids = defaultdict(list)
    for i in range(
        6, 209
    ):  # range(6,8): 3-node graphs, range(13,19): 4-node graphs, range(29,53): 5-node graphs
        g = nx.graph_atlas(i)
        if nx.is_connected(g):
            query_ids[len(g)].append(i)
        if len(g) > max(query_size):
            break

    return_ids = []

    for size, ids in query_ids.items():
        if size in query_size:
            return_ids.extend(ids)

    return return_ids


def SymmetricFactor(graph: nx.Graph, node_feat_key: str = None) -> int:
    """
    input: graph that need to compute symmetric factor, the node feature key that need to be considered
    output: symmetric fator, which is the proportion of the number of mapping and the number of pattern
    """
    maps = GenVMap(graph, graph, node_feat_key)
    return len(maps)


def GenVMap(
    subgraph: nx.Graph, graph: nx.Graph, node_feat_key: str = None
) -> list[map]:
    """
    input: subgraph and graph in nx.graph
    output: vmaps, map from nodes of subgraph to that of graph
    """
    # node_match = (lambda x, y: all([all(x[key] == y[key]) for key in node_feat_key])) if node_feat_key is not None else None # given that node_feat_key is a list of keys
    node_match = (
        (lambda x, y: x[node_feat_key] == y[node_feat_key])
        if node_feat_key is not None
        else None
    )
    GraphMatcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, subgraph, node_match=node_match
    )
    SBM_iter = GraphMatcher.subgraph_isomorphisms_iter()
    maps = [dict(zip(map.values(), map.keys())) for map in SBM_iter]
    return maps


def load_data(
    dataset_name: str, n_neighborhoods=-1, transform: list = None, train_split=0.6
):
    # make dir data if not exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # find "train" or "test" in the dataset name
    if "train" in dataset_name:
        dataset_split = "train"
        dataset_name = dataset_name.replace("_train", "")
    elif "test" in dataset_name:
        dataset_split = "test"
        dataset_name = dataset_name.replace("_test", "")
    else:
        dataset_split = None

    save_dir = "data/" + dataset_name

    # find if the index of the nodes in dataset is sorted by degree
    if "_relabelByDegree" in dataset_name:
        if transform is not None:
            transform.append(Relabel(mode="decreasing_degree"))
        else:
            transform = [Relabel(mode="decreasing_degree")]
        dataset_name = dataset_name.replace("_relabelByDegree", "")

    # combine transform
    if transform is not None:
        transform = pyg.transforms.Compose(transform)

    # find dataset in the data folder
    if dataset_name == "ENZYMES":
        dataset = TUDataset(root=save_dir, name="ENZYMES", transform=transform)
    elif dataset_name == "COX2":
        dataset = TUDataset(root=save_dir, name="COX2", transform=transform)
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root=save_dir, name="CiteSeer", transform=transform)
    elif dataset_name == "MUTAG":
        # dataset = Entities(root="data/MUTAG", name="MUTAG", transform= transform)
        dataset = TUDataset(root=save_dir, name="MUTAG", transform=transform)
    elif dataset_name == "Cora":
        dataset = Planetoid(root=save_dir, name="Cora", transform=transform)
    elif dataset_name == "P2P":
        dataset = P2P(root=save_dir, transform=transform)
    elif dataset_name == "Astro":
        dataset = Astro(root=save_dir, transform=transform)
    elif dataset_name == "REDDIT-BINARY":
        dataset = TUDataset(root=save_dir, name="REDDIT-BINARY", transform=transform)
    elif dataset_name == "arXiv":
        dataset = PygNodePropPredDataset(
            root=save_dir, name="ogbn-arxiv", transform=transform
        )
    elif dataset_name == "ZINC":
        raise NotImplementedError
        # dataset = MoleculeDataset(root='data/ZINC', name='ZINC', transform= transform)
    elif dataset_name == "IMDB-BINARY":
        dataset = TUDataset(root=save_dir, name="IMDB-BINARY", transform=transform)
    elif dataset_name.split("_")[0] == "syn":
        min_size = 5
        max_size = 41
        dataset = SyntheticDataset(
            min_size=min_size,
            max_size=max_size,
            graph_num=int(dataset_name.split("_")[1]),
            root="data/{}".format(dataset_name),
            transform=transform,
        )
    else:
        print(dataset_name)
        raise NotImplementedError

    # TODO: support to define train/test/valid split
    if dataset_split is None:
        return dataset
    elif dataset_split == "train":
        dataset = [g for g in dataset]
        random.seed(0)
        random.shuffle(dataset)
        return dataset[: int(len(dataset) * train_split)]
    elif dataset_split == "test":
        dataset = [g for g in dataset]
        random.seed(0)
        random.shuffle(dataset)
        return dataset[int(len(dataset) * train_split) :]
    else:
        print(dataset_name, " does not confirms to the naming convention")
        raise NotImplementedError

    return dataset


def graph_count(query: nx.Graph, target: nx.Graph):
    maps = GenVMap(query, target)
    return len(maps)


def core_trueCount_anchor(arg_dict):
    query = arg_dict["query"]
    graph_data = arg_dict["graph_data"]
    anchor_node = arg_dict["anchor_node"]
    target = pyg.utils.convert.to_networkx(graph_data, to_undirected=True)
    # target = graph_data

    maps = GenVMap(query, target)
    match_anchor_nodes = {vmap[anchor_node] for vmap in maps}
    return len(match_anchor_nodes)


def true_count_anchor(query: nx.Graph, dataset, num_worker: int):
    for node in query.nodes():
        if query.nodes[node]["anchor"] == 1:
            anchor_node = node
            break
    arg_dict_list = [
        {"query": query, "anchor_node": anchor_node, "graph_data": graph_data}
        for graph_data in dataset
    ]
    # arg_dict_list = [{'query':query, 'anchor_node':anchor_node, 'graph_data':graph_data} for graph_data in neighs]

    with mp.Pool(num_worker) as p:
        num_match_list = p.map(core_trueCount_anchor, arg_dict_list)
    num_match = sum(num_match_list)
    return num_match


def count_canonical(query: nx.Graph, target: nx.Graph, symmetry_factor=1) -> int:
    """
    input: query graph, target graph, symmetry_factor if there is any
    output: number of pattern
    """
    canonical_node = max(target.nodes())

    maps = GenVMap(query, target, relabel=True)
    canonical_maps = []
    for map in maps:
        if canonical_node in map.values():
            canonical_maps.append(map)
    count = len(canonical_maps) / symmetry_factor
    ### debug purpose
    # try:
    #     assert count%1 == 0
    # except AssertionError:
    #     print(count)
    ### end debug
    return int(count)


def count_graphlet(query: nx.Graph, target: nx.Graph, symmetry_factor=1) -> int:
    """
    input: query graph, target graph, symmetry_factor if there is any
    output: number of pattern
    """

    maps = GenVMap(query, target, relabel=True)
    count = len(maps) / symmetry_factor
    ### debug purpose
    try:
        assert count % 1 == 0
    except AssertionError:
        print(count)
    ### end debug
    return int(count)


def count_canonical_mp(query: nx.Graph, targets, num_worker: int, from_pyg=False):
    """
    input: query graph, iterable target graphs, numworker
    output: number of canonical count of targets
    """
    if from_pyg:
        arg_tuple = []
        for i in range(len(targets)):
            target = targets[i]
            arg_tuple.append(
                (query, pyg.utils.convert.to_networkx(target, to_undirected=True))
            )
        arg_tuple = tuple(arg_tuple)
    else:
        arg_tuple = tuple((query, target) for target in targets)

    with mp.Pool(num_worker) as pool:
        num_match_list = pool.starmap(count_canonical, arg_tuple)
    return num_match_list


def k_neigh(G: nx.Graph, start_node, k):
    neighs = set([start_node])
    fronts = set([start_node])
    for l in range(k):
        add_node = set()
        for n in fronts:
            add_node.update(G.neighbors(n))
        fronts = add_node.difference(neighs)
        neighs = neighs.union(fronts)
    return list(neighs)


def k_neigh_canonical(G: nx.Graph, start_node, k):
    neighs = set([start_node])
    fronts = set([start_node])
    for l in range(k):
        add_node = set()
        for n in fronts:
            add_node.update([n for n in G.neighbors(n) if n <= start_node])
        fronts = add_node.difference(neighs)
        neighs = neighs.union(fronts)
    return list(neighs)


def get_neigh_canonical(graph, node, radius: int):
    """
    input: target graph, canonical node, radius of sampling(k-hop)
    output: neighborhoods with 'node_feature'
    """
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)

    start_node = node
    neigh = graph.subgraph(
        [node for node in k_neigh_canonical(graph, start_node, radius)]
    )
    for component in nx.connected_components(neigh):
        if start_node in component:
            neigh = neigh.subgraph(component).copy()
            break
    for node in neigh.nodes:
        neigh.nodes[node]["node_feature"] = torch.zeros(1)
    neigh.nodes[start_node]["node_feature"] = torch.ones(1)
    return neigh


def get_neigh_hetero(graph, node, radius: int):
    """
    input: target graph, canonical node, radius of sampling(k-hop)
    output: neighborhoods with 'node_feature'
    """
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)

    start_node = node
    neigh = graph.subgraph(
        [node for node in k_neigh(graph, start_node, radius) if node <= start_node]
    )
    for component in nx.connected_components(neigh):
        if start_node in component:
            neigh = neigh.subgraph(component).copy()
            break
    for node in neigh.nodes:
        neigh.nodes[node]["type"] = "count"
        # neigh.nodes[node]['feat'] = torch.zeros(1)
    neigh.nodes[start_node]["type"] = "canonical"
    # assert(max(nx.shortest_path_length(neigh, start_node).values()) <= radius) will be false, use k_neigh_canonical to make sure it's right
    return neigh


def sample_neigh_canonical(graphs, radius: int):
    """
    input: graphs, radius of sampling(k-hop)
    output: neighborhoods with 'node_feature'
    """
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    idx = stats.rv_discrete(values=(np.arange(len(graphs)), ps)).rvs()
    # graph = random.choice(graphs)
    graph = graphs[idx]
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)
    while True:
        start_node = random.choice(list(graph.nodes))
        # neigh = nx.subgraph(graph, [node for node in nx.single_source_shortest_path_length(graph, start_node, radius) if node<=start_node]).copy()
        neigh = graph.subgraph(
            [node for node in k_neigh(graph, start_node, radius) if node <= start_node]
        )
        for component in nx.connected_components(neigh):
            if start_node in component:
                neigh = neigh.subgraph(component).copy()
                break
        if len(neigh.edges) == 0:
            continue
        for node in neigh.nodes:
            neigh.nodes[node]["node_feature"] = torch.zeros(1)
        neigh.nodes[start_node]["node_feature"] = torch.ones(1)
        return neigh


def sample_graphlet(graphs, *args, **kwargs):
    """
    input: graphs
    output: graphs(neighborhoods) without 'node_feature'
    """
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    idx = stats.rv_discrete(values=(np.arange(len(graphs)), ps)).rvs()
    # graph = random.choice(graphs)
    graph = graphs[idx]
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)
    return graph


class SyntheticDataset(InMemoryDataset):
    """
    dataset generated with mixed generators
    """

    def __init__(
        self,
        min_size: int = 4,
        max_size: int = 40,
        graph_num: int = 128,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.graph_num = graph_num

        self.name = "Synthetic_size_min_{:d}_max_{:d}_graph_num_{:d}".format(
            self.min_size, self.max_size, self.graph_num
        )
        self.sizes = [s for s in range(self.min_size + 1, self.max_size + 1)]

        # save as InMemoryDataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """
        A list of files in the raw_dir which needs to be found in order to skip the download.
        """
        edgelist_file_name = "{}_edgelist.txt".format(self.name)
        graph_indicator_file_name = "{}_graph_indicator.txt".format(self.name)
        return [
            edgelist_file_name,
            graph_indicator_file_name,
        ]  # can be used by calling self.raw_paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        """
        return [
            "{}_data.pt".format(self.name)
        ]  # can be used by calling self.processed_paths

    def gen_data_loaders(
        self, size, batch_size, train=True, use_distributed_sampling=False
    ):
        loader = []

        dataset = combined_syn.get_dataset(
            "graph", size, np.arange(self.min_size + 1, self.max_size + 1)
        )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            if use_distributed_sampling
            else None
        )
        loader = TorchDataLoader(
            dataset,
            collate_fn=Batch.collate([]),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
        )

        return loader

    def download(self):
        """
        Generate raw data and save to disk.
        """
        # define synthetic generators
        self.generator = combined_syn.get_generator(
            np.arange(self.min_size + 1, self.max_size + 1)
        )

        # generate graphs
        dataset = self.gen_data_loaders(
            self.graph_num, self.graph_num, train=True, use_distributed_sampling=False
        )
        dataset = next(iter(dataset)).G

        # add trival graphs of size 2 and 3 with prob 1E-3
        # for i in range(ceil(1E-3*float(self.graph_num))):
        #     dataset += [nx.graph_atlas(i) for i in (3,6,7)]

        random.shuffle(dataset)

        # merge all networkx graphs into one large graph
        init_node = 0
        for i in range(len(dataset)):
            dataset[i] = nx.convert_node_labels_to_integers(
                dataset[i], ordering="sorted", first_label=init_node
            )
            init_node += len(dataset[i])

        # save edgelist
        edgelist_file_name = "{}_edgelist.txt".format(self.name)
        edgelist_file_path = os.path.join(self.raw_dir, edgelist_file_name)
        with open(edgelist_file_path, "w") as f:
            f.write(
                "# {:d} {:d}\n".format(
                    sum([len(g.nodes) for g in dataset]),
                    sum([len(g.edges) for g in dataset]),
                )
            )
            for i in range(len(dataset)):
                for edge in dataset[i].edges:
                    f.write("{} {}\n".format(edge[0], edge[1]))

        # save graph indicator
        graph_indicator_file_name = "{}_graph_indicator.txt".format(self.name)
        graph_indicator_file_path = os.path.join(
            self.raw_dir, graph_indicator_file_name
        )
        with open(graph_indicator_file_path, "w") as f:
            f.write("# {:d}\n".format(len(dataset)))
            for i in range(len(dataset)):
                f.write("{:d}\n".format(len(dataset[i].edges)))

    def process(self):
        # read graph indicator
        graph_indicator_edge_num = []
        with open(self.raw_paths[1], "rt") as f:
            for line in f:
                if not line.startswith("#"):
                    graph_indicator_edge_num.append(int(line.strip("\n")))

        dataset_nx = [
            nx.Graph(directed=False) for _ in range(len(graph_indicator_edge_num))
        ]

        # read edgelist
        gid = 0
        eid = 0
        edge_list = []
        with open(self.raw_paths[0], "rt") as f:
            for line in f:
                if not line.startswith("#"):
                    splitted = line.strip("\n").split()
                    from_node = int(splitted[0])
                    to_node = int(splitted[1])
                    edge_list.append((from_node, to_node))
                    eid += 1
                    if eid == graph_indicator_edge_num[gid]:
                        dataset_nx[gid].add_edges_from(edge_list)
                        edge_list = []
                        gid += 1
                        eid = 0

        data_list = [pyg.utils.from_networkx(nx_graph) for nx_graph in dataset_nx]

        # init data_list with empty x
        for i in range(len(data_list)):
            data_list[i].x = torch.zeros(data_list[i].num_nodes, 1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class P2P(InMemoryDataset):
    """
    dataset from http://snap.stanford.edu/data/p2p-Gnutella04.txt.gz
    """

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """
        A list of files in the raw_dir which needs to be found in order to skip the download.
        """
        return ["p2p-Gnutella04.txt.gz"]  # can be used by calling self.raw_paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        """
        return ["data.pt"]

    def download(self):
        """
        Downloads raw data into raw_dir.
        """
        url = "http://snap.stanford.edu/data/p2p-Gnutella04.txt.gz"
        download_url(url, self.raw_dir)

    def process(self):
        """
        Processes raw data and saves it into the processed_dir.
        """
        edge_list = []
        with gzip.open(self.raw_paths[0], "rt") as f:
            for line in f:
                if not line.startswith("#"):
                    splitted = line.strip("\n").split()

                    from_node = int(splitted[0])
                    to_node = int(splitted[1])

                    edge_list.append([from_node, to_node])

        nx_graph = nx.Graph(directed=False)
        nx_graph.add_edges_from(edge_list)

        data_list = [pyg.utils.from_networkx(nx_graph)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Astro(InMemoryDataset):
    """
    dataset from http://snap.stanford.edu/data/ca-AstroPh.html
    """

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """
        A list of files in the raw_dir which needs to be found in order to skip the download.
        """
        return ["ca-AstroPh.txt.gz"]  # can be used by calling self.raw_paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        """
        return ["data.pt"]

    def download(self):
        """
        Downloads raw data into raw_dir.
        """
        url = "https://snap.stanford.edu/data/ca-AstroPh.txt.gz"
        download_url(url, self.raw_dir)

    def process(self):
        """
        Processes raw data and saves it into the processed_dir.
        """
        edge_list = []
        with gzip.open(self.raw_paths[0], "rt") as f:
            for line in f:
                if not line.startswith("#"):
                    splitted = line.strip("\n").split()

                    from_node = int(splitted[0])
                    to_node = int(splitted[1])

                    edge_list.append([from_node, to_node])

        nx_graph = nx.Graph(directed=False)
        nx_graph.add_edges_from(edge_list)

        data_list = [pyg.utils.from_networkx(nx_graph)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def batch_nx_graphs(graphs, anchors=None):
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    return batch


if __name__ == "__main__":
    # from common.utils import batch_nx_graphs

    # query = nx.Graph(directed = False)
    # query.add_edges_from([(0,1),(1,2)])

    # target = nx.Graph()
    # with open("/home/futy18/data/graph_data/greedyGather_edge_list/astro.txt", "r") as f:
    #     for row in f:
    #         if not row.startswith("#"):
    #             a, b = row.split(" ")
    #             target.add_edge(int(a), int(b))

    # cnt = graph_count(query, target)
    # print(cnt)

    """
    query = nx.Graph(directed = False)
    query.add_edges_from([(0,1),(1,2),(2,3)])

    len_neighbor = nx.diameter(graph)
    count = 0
    for node in graph.nodes:
        symmetry_factor = SymmetricFactor(query)
        target = get_neigh_canonical(graph, node, len_neighbor)
        count += count_canonical(query, target, symmetry_factor= symmetry_factor)
    print(count)
    """

    """
    subgraph1 = nx.Graph(directed = False)
    subgraph1.add_edges_from([(0,5),(0,2),(2,5)])

    neigh = count_canonical(subgraph1, graph, symmetry_factor=SymmetricFactor(graph))
    # neigh = count_canonical(graph, graph)
    # graph_pattern = {frozenset(map.values()) for map in maps}
    """

    print("done")
