import csv
import os
import os.path as osp
import pickle
import time
from itertools import permutations, product

import networkx as nx
import numpy as np
import torch
import torch.utils.data
import torch_geometric as pyg
from scipy import sparse as sp
from torch.utils.data import DataLoader

# from dgl.data.utils import Subset, load_graphs
from tqdm import tqdm

from subgraph_counting.gnn_model import LRP_GraphEmbModule


# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split, "r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [self.data[i] for i in data_idx[0]]

            assert (
                len(self.data) == num_graphs
            ), "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"

        """
        data is a list of Molecule dict objects with following attributes

          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        print(
            "preparing %d graphs for the %s set..."
            % (self.num_graphs, self.split.upper())
        )

        for molecule in self.data:
            node_features = molecule["atom_type"].long()

            adj = molecule["bond_type"]
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule["num_atom"])
            g.ndata["feat"] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata["feat"] = edge_features

            self.graph_lists.append(g)
            self.graph_labels.append(molecule["logP_SA_cycle_normalized"])

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get the idx^th sample.
        Parameters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field
            And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name="Zinc"):
        t0 = time.time()
        self.name = name

        self.num_atom_type = (
            28  # known meta-info about the zinc dataset; can be calculated as well
        )
        self.num_bond_type = (
            4  # known meta-info about the zinc dataset; can be calculated as well
        )

        data_dir = "./data/molecules"

        if self.name == "ZINC-full":
            data_dir = "./data/molecules/zinc_full"
            self.train = MoleculeDGL(data_dir, "train", num_graphs=220011)
            self.val = MoleculeDGL(data_dir, "val", num_graphs=24445)
            self.test = MoleculeDGL(data_dir, "test", num_graphs=5000)
        else:
            self.train = MoleculeDGL(data_dir, "train", num_graphs=10000)
            self.val = MoleculeDGL(data_dir, "val", num_graphs=1000)
            self.test = MoleculeDGL(data_dir, "test", num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time() - t0))


def self_loop(g):
    """
    Utility function only, to be used only when necessary as per user self_loop flag
    : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


    This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata["feat"] = g.ndata["feat"]

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata["feat"] = torch.zeros(new_g.number_of_edges())
    return new_g


def positional_encoding(g, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata["pos_enc"] = torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    return g


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        """
        Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = "data/molecules/"
        with open(data_dir + name + ".pkl", "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print(
            "train, test, val sizes :", len(self.train), len(self.test), len(self.val)
        )
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        # tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        # tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        # snorm_n = torch.cat(tab_snorm_n).sqrt()
        # tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        # tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        # snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        # tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        # tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        # snorm_n = tab_snorm_n[0][0].sqrt()

        # batched_graph = dgl.batch(graphs)

        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i.
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack(
                [zero_adj for j in range(self.num_atom_type + self.num_bond_type)]
            )
            adj_with_edge_feat = torch.cat(
                [adj.unsqueeze(0), adj_with_edge_feat], dim=0
            )

            us, vs = g.edges()
            for idx, edge_label in enumerate(g.edata["feat"]):
                adj_with_edge_feat[edge_label.item() + 1 + self.num_atom_type][us[idx]][
                    vs[idx]
                ] = 1

            for node, node_label in enumerate(g.ndata["feat"]):
                adj_with_edge_feat[node_label.item() + 1][node][node] = 1

            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)

            return None, x_with_edge_feat, labels

        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack(
                [zero_adj for j in range(self.num_atom_type)]
            )
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_label in enumerate(g.ndata["feat"]):
                adj_no_edge_feat[node_label.item() + 1][node][node] = 1

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1.0 / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [
            positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists
        ]
        self.val.graph_lists = [
            positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists
        ]
        self.test.graph_lists = [
            positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists
        ]


def seq_generate_easy(E_list, start_node, length=4):
    all_perm = permutations(E_list[start_node], min(length, len(E_list[start_node])))
    return [[start_node] + list(p) for p in all_perm]


def seq_generate_deep(E_list, start_node, depth=1, node_per_layer=1):
    current_seq_set = [[[], [start_node]]]
    # prev, this_depth
    current_depth = 0

    while current_depth < depth:
        new_seq_set = []
        for seq in current_seq_set:
            prev, this_depth = seq

            new_perm_set = [[]]
            for node in this_depth:
                new_new_perm_set = []

                for new_perm in new_perm_set:
                    new_node_children = list(
                        set(E_list[node]) - set(prev) - set(this_depth) - set(new_perm)
                    )
                    all_perm = permutations(
                        new_node_children, min(node_per_layer, len(new_node_children))
                    )

                    for p in all_perm:
                        new_new_perm_set.append(new_perm + list(p))

                new_perm_set = new_new_perm_set

            for p in new_perm_set:
                new_seq_set.append([prev + this_depth, p])
        current_seq_set = new_seq_set
        current_depth += 1
    seq_set = [p + q for p, q in current_seq_set]

    return seq_set


def seq_to_sp_indx(graph, one_perm, subtensor_length):
    edge_index = graph.edge_index

    dim_dict = {node: i for i, node in enumerate(one_perm)}

    node_to_length_indx_row = [i + i * subtensor_length for i in range(len(one_perm))]
    node_to_length_indx_col = one_perm

    product_one_perm = list(product(one_perm, one_perm))
    query_edge_id_src, query_edge_id_end = [edge[0] for edge in product_one_perm], [
        edge[1] for edge in product_one_perm
    ]

    # query_edge_result = graph.edge_ids(query_edge_id_src, query_edge_id_end, return_uv = True)
    query_edge_result = [[], [], []]
    for u, v in zip(query_edge_id_src, query_edge_id_end):
        match = (edge_index == torch.tensor([u, v]).view(2, 1)).all(dim=0)
        if match.any():
            query_edge_result[0].append(u)
            query_edge_result[1].append(v)
            query_edge_result[2].append(
                match.nonzero().item()
            )  # ALERT: only one match should exits

    edge_to_length_indx_row = [
        int(dim_dict[src] * subtensor_length + dim_dict[end])
        for src, end, _ in zip(*query_edge_result)
    ]
    edge_to_length_indx_col = [int(edge_id) for edge_id in query_edge_result[2]]

    # ALERT: edge_to_length_indx_row and edge_to_length_indx_col are not generated in the code
    # edge_to_length_indx_row = [int(dim_dict[src] * subtensor_length + dim_dict[end]) for src, end in zip(query_edge_id_src, query_edge_id_end)]
    # edge_to_length_indx_col = [np.NaN for _ in range(len(edge_to_length_indx_row))] #bug to fix

    return [
        node_to_length_indx_row,
        node_to_length_indx_col,
        edge_to_length_indx_row,
        edge_to_length_indx_col,
    ]


class LRP_Dataset(object):
    def __init__(
        self,
        dataset_name,
        graphs,
        labels,
        lrp_save_path="dataset",
        lrp_depth=1,
        subtensor_length=4,
        lrp_width=3,
    ):
        super(LRP_Dataset, self).__init__()
        """
        graphs is an iterable object, each element is a pyg graph
        labels is a list of tensor, indicating the ground truth for each graph
        """
        # load data to self.graphs
        self.graphs = graphs
        self.labels = labels

        self.subtensor_length = subtensor_length
        self.lrp_depth = lrp_depth
        self.lrp_width = lrp_width

        assert self.subtensor_length == self.lrp_depth * self.lrp_width + 1

        # self.num_tasks = 1

        self.output_length = int(subtensor_length**2)
        self.lrp_save_path = lrp_save_path
        self.lrp_save_file = (
            "lrp_"
            + dataset_name
            + "_dep"
            + str(lrp_depth)
            + "_wid"
            + str(lrp_width)
            + "_len"
            + str(subtensor_length)
        )
        self.lrp_egonet_seq = np.array([])  # load from file or generate
        self.load_lrp()

        # self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        # self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well

    def load_lrp(self):
        print("Trying to load LRP!")
        lrp_save_file = osp.join(self.lrp_save_path, self.lrp_save_file + ".npz")

        if os.path.exists(lrp_save_file):
            print("LRP file exists!")
            read_file = np.load(lrp_save_file, allow_pickle=True)
            read_lrp = read_file["a"]
            if len(read_lrp) == len(self.graphs):
                print("LRP file format correct! ", self.output_length)
                self.lrp_egonet_seq = read_lrp
            else:
                print("LRP file format WRONG!")
                self.LRP_preprocess()
        else:
            print("LRP file does not exist!")
            self.LRP_preprocess()

    def save_lrp(self):
        print("Saving LRP to ", osp.join(self.lrp_save_path, self.lrp_save_file))
        np.savez(
            osp.join(self.lrp_save_path, self.lrp_save_file), a=self.lrp_egonet_seq
        )
        print("Saving LRP FINISHED!")

    def LRP_preprocess(self):
        print("Preprocessing LRP!")
        if len(self.lrp_egonet_seq) == 0:
            self.lrp_egonet_seq = []
            for g in tqdm(self.graphs):
                self.lrp_egonet_seq.append(
                    lrp_helper(
                        g,
                        subtensor_length=self.subtensor_length,
                        lrp_depth=self.lrp_depth,
                        lrp_width=self.lrp_width,
                    )
                )

            # self.lrp_egonet_seq = np.array(self.lrp_egonet_seq)
            if len(self.lrp_egonet_seq) == len(self.graphs):
                print("LRP generated with correct format")
            else:
                print("LRP generated WRONG: PLEASE CHECK!")
                print(len(self.lrp_egonet_seq), len(self.graphs))
                exit()
            self.save_lrp()

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.lrp_egonet_seq[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.lrp_egonet_seq[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            "Only integers and long are valid "
            "indices (got {}).".format(type(idx).__name__)
        )

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return "{}({})".format(self.__class__.__name__, len(self))


def nx_edge_list(graph: nx.Graph):
    edge_list = []
    for n in graph.nodes:
        edge_list.append(np.array(list(graph.neighbors(n))))
    return edge_list


def lrp_helper(graph: pyg.data.Data, subtensor_length=4, lrp_depth=1, lrp_width=3):
    """
    \'helper\' function
    graph is a pyg graph with properties:
    graph.node_feature (num_nodes, num_node_features)
    """
    nx_graph = pyg.utils.to_networkx(graph, to_undirected=True)

    num_of_nodes = len(nx_graph)
    graph_edge_list = nx_edge_list(nx_graph)

    egonet_seq_graph = []

    for i in range(num_of_nodes):
        # this_node_perms = seq_generate(graph_Elist, i, 1, split_level = False)
        if lrp_depth == 1:
            this_node_perms = seq_generate_easy(
                graph_edge_list, start_node=i, length=subtensor_length - 1
            )
        else:
            this_node_perms = seq_generate_deep(
                graph_edge_list, start_node=i, depth=lrp_depth, node_per_layer=lrp_width
            )
        this_node_egonet_seq = []

        for perm in this_node_perms:
            this_node_egonet_seq.append(seq_to_sp_indx(graph, perm, subtensor_length))
        # this_node_egonet_seq = np.array(this_node_egonet_seq)
        egonet_seq_graph.append(this_node_egonet_seq)

    # print(egonet_seq_graph)
    # print(graph.in_degrees(list(range(graph.number_of_nodes()))))

    # egonet_seq_graph = np.array(egonet_seq_graph)

    return egonet_seq_graph


def build_batch_graph_node_to_perm_times_length(graphs, lrp_egonet):
    """
    graphs: list of DGLGraph
    lrp_egonet: list of egonet of graph, dim: #graphs x #nodes x #perms x
                 (sparse index for length x #nodes or #edges)
    """
    list_num_nodes_in_graphs = [g.number_of_nodes() for g in graphs]
    sum_num_nodes_before_graphs = [
        sum(list_num_nodes_in_graphs[:i]) for i in range(len(graphs))
    ]
    list_num_edges_in_graphs = [g.number_of_edges() for g in graphs]
    sum_num_edges_before_graphs = [
        sum(list_num_edges_in_graphs[:i]) for i in range(len(graphs))
    ]

    node_to_perm_length_indx_row = []
    node_to_perm_length_indx_col = []
    edge_to_perm_length_indx_row = []
    edge_to_perm_length_indx_col = []

    sum_row_number = 0

    for i, g_egonet in enumerate(lrp_egonet):
        for n_egonet in g_egonet:
            for perm in n_egonet:
                node_to_perm_length_indx_col.extend(
                    np.array(perm[1]) + sum_num_nodes_before_graphs[i]
                )
                node_to_perm_length_indx_row.extend(np.array(perm[0]) + sum_row_number)

                edge_to_perm_length_indx_col.extend(
                    np.array(perm[3]) + sum_num_edges_before_graphs[i]
                )
                edge_to_perm_length_indx_row.extend(np.array(perm[2]) + sum_row_number)

                sum_row_number += 16

    node_to_perm_length_size_row = sum_row_number
    node_to_perm_length_size_col = sum(list_num_nodes_in_graphs)
    edge_to_perm_length_size_row = sum_row_number
    edge_to_perm_length_size_col = sum(list_num_edges_in_graphs)

    data1 = np.ones(
        (
            len(
                node_to_perm_length_indx_col,
            )
        )
    )
    node_to_perm_length_sp_matrix = csr_matrix(
        (data1, (node_to_perm_length_indx_row, node_to_perm_length_indx_col)),
        shape=(node_to_perm_length_size_row, node_to_perm_length_size_col),
    )

    data2 = np.ones(
        (
            len(
                edge_to_perm_length_indx_col,
            )
        )
    )
    edge_to_perm_length_sp_matrix = csr_matrix(
        (data2, (edge_to_perm_length_indx_row, edge_to_perm_length_indx_col)),
        shape=(edge_to_perm_length_size_row, edge_to_perm_length_size_col),
    )

    return node_to_perm_length_sp_matrix, edge_to_perm_length_sp_matrix


def build_batch_graph_node_to_perm_times_length_index_form(
    graphs: pyg.data.Data, lrp_egonet, subtensor_length=4
):
    """
    graphs: list of DGLGraph
    lrp_egonet: list of egonet of graph, dim: #graphs x #nodes x #perms x
                 (sparse index for length x #nodes or #edges)
    """
    list_num_nodes_in_graphs = [g.node_feature.shape[0] for g in graphs]
    sum_num_nodes_before_graphs = [
        sum(list_num_nodes_in_graphs[:i]) for i in range(len(graphs))
    ]
    list_num_edges_in_graphs = [g.edge_index.shape[1] for g in graphs]
    sum_num_edges_before_graphs = [
        sum(list_num_edges_in_graphs[:i]) for i in range(len(graphs))
    ]

    node_to_perm_length_indx_row = []
    node_to_perm_length_indx_col = []
    edge_to_perm_length_indx_row = []
    edge_to_perm_length_indx_col = []

    sum_row_number = 0

    for i, g_egonet in enumerate(lrp_egonet):
        for n_egonet in g_egonet:
            for perm in n_egonet:
                node_to_perm_length_indx_col.extend(
                    np.array(perm[1]) + sum_num_nodes_before_graphs[i]
                )
                node_to_perm_length_indx_row.extend(np.array(perm[0]) + sum_row_number)

                edge_to_perm_length_indx_col.extend(
                    np.array(perm[3]) + sum_num_edges_before_graphs[i]
                )
                edge_to_perm_length_indx_row.extend(np.array(perm[2]) + sum_row_number)

                sum_row_number += int(subtensor_length**2)

    node_to_perm_length_size_row = sum_row_number
    node_to_perm_length_size_col = sum(list_num_nodes_in_graphs)
    edge_to_perm_length_size_row = sum_row_number
    edge_to_perm_length_size_col = sum(list_num_edges_in_graphs)

    data1 = np.ones(
        (
            len(
                node_to_perm_length_indx_col,
            )
        )
    )

    data2 = np.ones(
        (
            len(
                edge_to_perm_length_indx_col,
            )
        )
    )

    return (
        np.array([node_to_perm_length_indx_row, node_to_perm_length_indx_col]),
        data1,
        node_to_perm_length_size_row,
        node_to_perm_length_size_col,
        np.array([edge_to_perm_length_indx_row, edge_to_perm_length_indx_col]),
        data2,
        edge_to_perm_length_size_row,
        edge_to_perm_length_size_col,
    )


def build_perm_pooling_sp_matrix_index_form(split_list, pooling="sum"):
    dim0, dim1 = len(split_list), sum(split_list)
    col = np.arange(dim1)
    row = np.array([i for i, count in enumerate(split_list) for j in range(count)])

    if pooling == "sum":
        data = np.ones((dim1,))
    elif pooling == "mean":
        data = np.array([1 / s for s in split_list for i in range(s)])
    else:
        assert False

    return np.stack([row, col]), data, dim0, dim1


def collate_lrp_dgl_light(samples):
    graphs, lrp_egonets, labels = map(list, zip(*samples))
    n_to_pl, e_to_pl = build_batch_graph_node_to_perm_times_length(graphs, lrp_egonets)
    batched_graph = dgl.batch(graphs)
    return (
        batched_graph,
        [len(node) for g in lrp_egonets for node in g],
        [n_to_pl, e_to_pl],
        torch.stack(labels),
    )


def collate_lrp_dgl_light_index_form_wrapper(subtensor_length=4):
    def collate_lrp_dgl_light_index_form(samples):
        graphs, lrp_egonets, labels = map(list, zip(*samples))
        egonet_to_perm_pl = build_batch_graph_node_to_perm_times_length_index_form(
            graphs, lrp_egonets, subtensor_length=subtensor_length
        )
        batched_graph = pyg.data.Batch.from_data_list(graphs)
        split_list = [len(node) for g in lrp_egonets for node in g]
        perm_pooling_matrix = build_perm_pooling_sp_matrix_index_form(
            split_list, "mean"
        )
        return (
            batched_graph,
            perm_pooling_matrix,
            egonet_to_perm_pl,
            torch.stack(labels, dim=0),
        )

    return collate_lrp_dgl_light_index_form


def LoadData(DATASET_NAME):
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == "ZINC":
        return MoleculeDataset(DATASET_NAME)


if __name__ == "__main__":
    from subgraph_counting.data import load_data
    from subgraph_counting.train import to_device

    device = "cuda"

    graphs = load_data("ENZYMES")
    graphs = [g for g in graphs]
    for g in graphs:
        g.node_feature = g.x
    labels = [torch.zeros(29) for i in range(len(graphs))]
    dataset = LRP_Dataset(
        "ENZYMES",
        graphs=graphs,
        labels=labels,
        lrp_save_path="subgraph_counting/workload/general/ENZYMES_gossip_n_query_29_all_hetero",
        lrp_depth=1,
        subtensor_length=4,
        lrp_width=3,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_lrp_dgl_light_index_form_wrapper(4),
        num_workers=1,
    )

    # print(dataloader)
    # for data in dataloader:
    #     print(data)
    queries_pyg = [
        pyg.utils.from_networkx(graph_atlas_plus(query_id))
        for query_id in [6, 7, 13, 14]
    ]

    for query_pyg in queries_pyg:
        query_pyg.node_feature = torch.zeros((query_pyg.num_nodes, 1), device=device)

    query_LRP = LRP_Dataset(
        "queries_" + str(len(queries_pyg)),
        graphs=queries_pyg,
        labels=[torch.zeros(1) for _ in range(len(queries_pyg))],
        lrp_save_path="/home/nfs_data/futy/repos/prime/GNN_Mining/2021Summer/tmp_folder",
        lrp_depth=1,
        subtensor_length=4,
        lrp_width=3,
    )

    query_loader = DataLoader(
        query_LRP,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_lrp_dgl_light_index_form_wrapper(4),
    )

    hyperparam_list = {
        # 'lrp_length': 16,
        # 'num_tasks': dataset.num_tasks,
        # 'lrp_in_dim': 13,
        # 'hid_dim': 13,
        # 'num_layers': 4,
        # 'lr': args.lr
        "lrp_length": int(4**2),
        "num_tasks": 1,
        "lrp_in_dim": 1,  # node feats dim
        "hid_dim": 128,
        "num_layers": 4,
        "bn": False,
        "lr": 0.001,
        "mlp": False,
        "alldegree": False,
    }

    model = LRP_GraphEmbModule(
        num_tasks=hyperparam_list["num_tasks"],
        lrp_length=hyperparam_list["lrp_length"],
        lrp_in_dim=hyperparam_list["lrp_in_dim"],
        hid_dim=hyperparam_list["hid_dim"],
        num_layers=hyperparam_list["num_layers"],
        bn=hyperparam_list["bn"],
        mlp=hyperparam_list["mlp"],
        alldegree=hyperparam_list["alldegree"],
        num_atom_type=hyperparam_list["lrp_in_dim"],
    ).to(device)

    for iter, lrp_data in tqdm(enumerate(query_loader)):
        lrp_data = to_device(lrp_data, device)
        # batch, pooling_matrix, sp_matrices, label = lrp_data
        # batch = batch.to(device)
        # pooling_matrix = [torch.LongTensor(pooling_matrix[0]).to(device), torch.FloatTensor(pooling_matrix[1]).to(device), pooling_matrix[2], pooling_matrix[3]]
        # n_to_perm = [torch.LongTensor(sp_matrices[0]).to(device), torch.FloatTensor(sp_matrices[1]).to(device), sp_matrices[2], sp_matrices[3]]
        # e_to_perm = [torch.LongTensor(sp_matrices[4]).to(device), torch.FloatTensor(sp_matrices[5]).to(device), sp_matrices[6], sp_matrices[7]]
        # degs = pyg.utils.degree(batch.edge_index[1,:]).type(torch.FloatTensor).to(device)

        emb = model(lrp_data)

        # print(emb)

    print("done")


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
