import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import gzip
import multiprocessing as mp
import random
from typing import Callable, List, Optional, Tuple, Union

import deepsnap as ds
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from common import combined_syn, feature_preprocess, utils
from common.data import DataSource
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph as DSGraph
from matplotlib import cm
from networkx.algorithms.operators.product import tensor_product
from networkx.classes.function import to_undirected
from networkx.convert import to_networkx_graph
from ogb.nodeproppred import PygNodePropPredDataset
from playground.lib.Anchor import GenVMap, SymmetricFactor
from sklearn.decomposition import PCA
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.datasets import Entities, Planetoid, TUDataset
from tqdm import tqdm


def load_data(dataset_name: str, n_neighborhoods= -1, transform= None):
    # make dir data if not exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # find dataset in the data folder
    if dataset_name == "ENZYMES":
        dataset = TUDataset(root="data/ENZYMES", name="ENZYMES", transform= transform)
    elif dataset_name == "COX2":
        dataset = TUDataset(root="data/COX2", name="COX2", transform= transform)
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data/CiteSeer", name="CiteSeer", transform= transform)
    elif dataset_name == "MUTAG":
        # dataset = Entities(root="data/MUTAG", name="MUTAG", transform= transform)
        dataset = TUDataset(root="data/MUTAG", name="MUTAG", transform= transform)
    elif dataset_name == "Cora":
        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=transform)
    elif dataset_name == "P2P":
        dataset = P2P(root='data/P2P', transform= transform)
    elif dataset_name == "Astro":
        dataset = Astro(root='data/Astro', transform= transform)
    elif dataset_name == 'REDDIT-BINARY':
        dataset = TUDataset(root='data/REDDIT-BINARY', name='REDDIT-BINARY', transform= transform)
    elif dataset_name == 'arXiv':
        dataset = PygNodePropPredDataset(root='data/arXiv', name='ogbn-arxiv', transform= transform)
    elif dataset_name == 'ZINC':
        raise NotImplementedError
        # dataset = MoleculeDataset(root='data/ZINC', name='ZINC', transform= transform)
    elif dataset_name == "syn":
        min_size = 5
        max_size = 41
        data_source = OTFSynCanonicalDataSource(min_size=min_size, max_size=max_size, canonical_anchored= True)
        # graph_num = n_neighborhoods
        graph_num = n_neighborhoods
        dataset = data_source.gen_data_loaders(graph_num, graph_num,
        train=True, use_distributed_sampling=False)
        dataset = next(iter(dataset)).G
        # add trival graphs
        for i in range(10):
            dataset += [nx.graph_atlas(i) for i in (3,6,7)]
        random.shuffle(dataset)
        if transform is not None:
            new_dataset = []
            for graph in dataset:
                new_dataset.append(transform(pyg.utils.from_networkx(graph)))
            dataset = new_dataset
        else:
            dataset = [pyg.utils.from_networkx(graph) for graph in dataset]
    else:
        print(dataset_name)
        raise NotImplementedError

    return dataset


def graph_count(query: nx.Graph, target: nx.Graph):
    maps = GenVMap(query, target)
    return len(maps)

def core_trueCount_anchor(arg_dict):
    query = arg_dict['query']
    graph_data = arg_dict['graph_data']
    anchor_node = arg_dict['anchor_node']
    target = pyg.utils.convert.to_networkx(graph_data, to_undirected= True)
    # target = graph_data

    maps = GenVMap(query, target)
    match_anchor_nodes = {vmap[anchor_node] for vmap in maps} 
    return len(match_anchor_nodes)

def true_count_anchor(query: nx.Graph, dataset, num_worker: int):
    for node in query.nodes():
        if query.nodes[node]['anchor'] == 1:
            anchor_node = node
            break
    arg_dict_list = [{'query':query, 'anchor_node':anchor_node, 'graph_data':graph_data} for graph_data in dataset]
    # arg_dict_list = [{'query':query, 'anchor_node':anchor_node, 'graph_data':graph_data} for graph_data in neighs]
    
    with mp.Pool(num_worker) as p:
        num_match_list = p.map(core_trueCount_anchor, arg_dict_list)
    num_match = sum(num_match_list)
    return num_match


def count_canonical(query: nx.Graph, target: nx.Graph, symmetry_factor=1) -> int:
    '''
    input: query graph, target graph, symmetry_factor if there is any
    output: number of pattern
    '''
    canonical_node = max(target.nodes())

    maps = GenVMap(query, target, relabel= True)
    canonical_maps = []
    for map in maps:
        if canonical_node in map.values():
            canonical_maps.append(map)
    count = len(canonical_maps)/symmetry_factor
    ### debug purpose
    # try:
    #     assert count%1 == 0
    # except AssertionError:
    #     print(count)
    ### end debug
    return int(count)

def count_graphlet(query: nx.Graph, target: nx.Graph, symmetry_factor=1) -> int:
    '''
    input: query graph, target graph, symmetry_factor if there is any
    output: number of pattern
    '''

    maps = GenVMap(query, target, relabel= True)
    count = len(maps)/symmetry_factor
    ### debug purpose
    try:
        assert count%1 == 0
    except AssertionError:
        print(count)
    ### end debug
    return int(count)



def count_canonical_mp(query: nx.Graph, targets, num_worker: int, from_pyg= False):
    '''
    input: query graph, iterable target graphs, numworker
    output: number of canonical count of targets
    '''
    if from_pyg:
        arg_tuple = []
        for i in range(len(targets)):
            target = targets[i]
            arg_tuple.append((query, pyg.utils.convert.to_networkx(target, to_undirected= True)))
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
            add_node.update([n for n in G.neighbors(n) if n<=start_node])
        fronts = add_node.difference(neighs)
        neighs = neighs.union(fronts)
    return list(neighs)

def get_neigh_canonical(graph, node, radius:int):
    '''
    input: target graph, canonical node, radius of sampling(k-hop)
    output: neighborhoods with 'node_feature'
    '''
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)

    start_node = node
    neigh = graph.subgraph([node for node in k_neigh_canonical(graph, start_node, radius)])
    for component in nx.connected_components(neigh):
        if start_node in component:
            neigh = neigh.subgraph(component).copy()
            break
    for node in neigh.nodes:
        neigh.nodes[node]['node_feature'] = torch.zeros(1)
    neigh.nodes[start_node]['node_feature'] = torch.ones(1)
    return neigh

def get_neigh_hetero(graph, node, radius:int):
    '''
    input: target graph, canonical node, radius of sampling(k-hop)
    output: neighborhoods with 'node_feature'
    '''
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)

    start_node = node
    neigh = graph.subgraph([node for node in k_neigh(graph, start_node, radius) if node<=start_node])
    for component in nx.connected_components(neigh):
        if start_node in component:
            neigh = neigh.subgraph(component).copy()
            break
    for node in neigh.nodes:
        neigh.nodes[node]['type'] = 'count'
        # neigh.nodes[node]['feat'] = torch.zeros(1)
    neigh.nodes[start_node]['type'] = 'canonical'
    # assert(max(nx.shortest_path_length(neigh, start_node).values()) <= radius) will be false, use k_neigh_canonical to make sure it's right
    return neigh

def sample_neigh_canonical(graphs, radius: int):
    '''
    input: graphs, radius of sampling(k-hop)
    output: neighborhoods with 'node_feature'
    '''
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    idx = stats.rv_discrete(values=(np.arange(len(graphs)), ps)).rvs()
    #graph = random.choice(graphs)
    graph = graphs[idx]
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)
    while True:
        start_node = random.choice(list(graph.nodes))
        # neigh = nx.subgraph(graph, [node for node in nx.single_source_shortest_path_length(graph, start_node, radius) if node<=start_node]).copy()
        neigh = graph.subgraph([node for node in k_neigh(graph, start_node, radius) if node<=start_node])
        for component in nx.connected_components(neigh):
            if start_node in component:
                neigh = neigh.subgraph(component).copy()
                break
        if len(neigh.edges)==0:
            continue
        for node in neigh.nodes:
            neigh.nodes[node]['node_feature'] = torch.zeros(1)
        neigh.nodes[start_node]['node_feature'] = torch.ones(1)
        return neigh

def sample_graphlet(graphs, *args, **kwargs):
    '''
    input: graphs
    output: graphs(neighborhoods) without 'node_feature'
    '''
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    idx = stats.rv_discrete(values=(np.arange(len(graphs)), ps)).rvs()
    #graph = random.choice(graphs)
    graph = graphs[idx]
    if type(graph) == pyg.data.data.Data:
        graph = pyg_utils.to_networkx(graph, to_undirected=True)
    return graph 

class OTFSynCanonicalDataSource(DataSource):
    """ On-the-fly generated synthetic data for training the subgraph model.

    At every iteration, new batch of graphs are generated with a pre-defined generator

    DeepSNAP transforms are used to generate the positive and negative examples.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, canonical_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.canonica_anchored = canonical_anchored
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

        self.sizes = [s for s in range(self.min_size + 1, self.max_size + 1)]

    def gen_nx(self):
        return self.generator.generate(size=random.choice(self.sizes))

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loader = []

        dataset = combined_syn.get_dataset("graph", size,
            np.arange(self.min_size + 1, self.max_size + 1))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()) if \
                use_distributed_sampling else None
        loader = TorchDataLoader(dataset,
            collate_fn=Batch.collate([]), batch_size=batch_size,
            sampler=sampler, shuffle=False)

        return loader

    def gen_batch(self, batch_target, batch_neg_target, query,
        train=True):
        
        augmenter = feature_preprocess.FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_neigh_canonical)
        neg_target = batch_neg_target
        # TODO: use hard negs
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
            int(len(neg_target.G) * 1/2)))
        #hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                if i not in hard_neg_idxs else g)
                for i, g in enumerate(neg_target.G)])
        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_canonical,
            hard_neg_idxs=hard_neg_idxs)
        if self.canonica_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                            else torch.zeros(1))
                return g
            neg_target = neg_target.apply_transform(add_anchor)
        pos_target = augmenter.augment(pos_target).to(utils.get_device())
        pos_query = augmenter.augment(pos_query).to(utils.get_device())
        neg_target = augmenter.augment(neg_target).to(utils.get_device())
        neg_query = augmenter.augment(neg_query).to(utils.get_device())
        #print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query


class P2P(InMemoryDataset):
    '''
    dataset from http://snap.stanford.edu/data/p2p-Gnutella04.txt.gz
    '''
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        '''
        A list of files in the raw_dir which needs to be found in order to skip the download.
        '''
        return ['p2p-Gnutella04.txt.gz'] # can be used by calling self.raw_paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        '''
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        '''
        return ['data.pt']

    def download(self):
        '''
        Downloads raw data into raw_dir.
        '''
        url= 'http://snap.stanford.edu/data/p2p-Gnutella04.txt.gz'
        download_url(url, self.raw_dir)

    def process(self):
        '''
        Processes raw data and saves it into the processed_dir.
        '''
        edge_list = []
        with gzip.open(self.raw_paths[0], 'rt') as f:
            for line in f:
                if not line.startswith("#"):
                    splitted = line.strip('\n').split()

                    from_node = int(splitted[0])
                    to_node   = int(splitted[1])

                    edge_list.append([from_node, to_node])

        nx_graph = nx.Graph(directed= False)
        nx_graph.add_edges_from(edge_list)

        data_list = [pyg.utils.from_networkx(nx_graph)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Astro(InMemoryDataset):
    '''
    dataset from http://snap.stanford.edu/data/ca-AstroPh.html
    '''
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        '''
        A list of files in the raw_dir which needs to be found in order to skip the download.
        '''
        return ['ca-AstroPh.txt.gz'] # can be used by calling self.raw_paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        '''
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        '''
        return ['data.pt']

    def download(self):
        '''
        Downloads raw data into raw_dir.
        '''
        url= 'https://snap.stanford.edu/data/ca-AstroPh.txt.gz'
        download_url(url, self.raw_dir)

    def process(self):
        '''
        Processes raw data and saves it into the processed_dir.
        '''
        edge_list = []
        with gzip.open(self.raw_paths[0], 'rt') as f:
            for line in f:
                if not line.startswith("#"):
                    splitted = line.strip('\n').split()

                    from_node = int(splitted[0])
                    to_node   = int(splitted[1])

                    edge_list.append([from_node, to_node])

        nx_graph = nx.Graph(directed= False)
        nx_graph.add_edges_from(edge_list)

        data_list = [pyg.utils.from_networkx(nx_graph)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
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



    '''
    query = nx.Graph(directed = False)
    query.add_edges_from([(0,1),(1,2),(2,3)])

    len_neighbor = nx.diameter(graph)
    count = 0
    for node in graph.nodes:
        symmetry_factor = SymmetricFactor(query)
        target = get_neigh_canonical(graph, node, len_neighbor)
        count += count_canonical(query, target, symmetry_factor= symmetry_factor)
    print(count)
    '''

    '''
    subgraph1 = nx.Graph(directed = False)
    subgraph1.add_edges_from([(0,5),(0,2),(2,5)])

    neigh = count_canonical(subgraph1, graph, symmetry_factor=SymmetricFactor(graph))
    # neigh = count_canonical(graph, graph)
    # graph_pattern = {frozenset(map.values()) for map in maps}
    '''

    print("done")
