# Combination of synthetic graph generators # (first used in subgraph matching and motif mining)

import logging
import networkx as nx
import numpy as np

import deepsnap.dataset as dataset

class ERGenerator(dataset.Generator):
    def __init__(self, sizes, p_alpha=1.3, m_scale=1, **kwargs):
        super(ERGenerator, self).__init__(sizes, **kwargs)
        self.p_alpha = p_alpha
        self.m_scale = m_scale

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        # p follows beta distribution with mean = log2(num_graphs) / num_graphs
        alpha = self.p_alpha
        mean = self.m_scale * np.log2(num_nodes) / num_nodes
        beta = alpha / mean - alpha
        p = np.random.beta(alpha, beta) # p follows beta distribution with mean = log2(num_nodes) / num_nodes
        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            p = np.random.beta(alpha, beta)
            graph = nx.gnp_random_graph(num_nodes, p)
        logging.debug('Generated {}-node E-R graphs with average p: {}'.format(
                num_nodes, mean))
        return graph

class WSGenerator(dataset.Generator):
    def __init__(self, sizes, density_alpha=1.3, 
            rewire_alpha=2, rewire_beta=2, m_scale=1, **kwargs):
        super(WSGenerator, self).__init__(sizes, **kwargs)
        self.density_alpha = density_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta
        self.m_scale = m_scale
    
    def generate(self, size=None):
        num_nodes = self._get_size(size)
        curr_num_graphs = 0

        density_alpha = self.density_alpha
        density_mean = self.m_scale*np.log2(num_nodes) / num_nodes
        density_beta = density_alpha / density_mean - density_alpha

        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta
        while curr_num_graphs < 1:
            k = int(np.random.beta(density_alpha, density_beta) * num_nodes)
            k = max(k, 2)
            p = np.random.beta(rewire_alpha, rewire_beta)
            try:
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
                curr_num_graphs += 1
            except:
                pass
        logging.debug('Generated {}-node W-S graph with average density: {}'.format(
                num_nodes, density_mean))
        return graph

class BAGenerator(dataset.Generator):
    def __init__(self, sizes, max_p=0.2, max_q=0.2, m_scale=2, **kwargs):
        super(BAGenerator, self).__init__(sizes, **kwargs)
        self.max_p = max_p
        self.max_q = max_q
        self.m_scale = m_scale

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(self.m_scale * np.log2(num_nodes))
        m = np.random.choice(max_m) + 1
        p = np.min([np.random.exponential(20), self.max_p])
        q = np.min([np.random.exponential(20), self.max_q])
        graph = nx.extended_barabasi_albert_graph(num_nodes, m, p, q)
        c = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(c).copy()
        logging.debug('Generated {}-node extended B-A graph with max m: {}'.format(num_nodes, max_m))
        return graph

class PowerLawClusterGenerator(dataset.Generator):
    def __init__(self, sizes, max_triangle_prob=0.5, m_scale=2, **kwargs):
        super(PowerLawClusterGenerator, self).__init__(sizes, **kwargs)
        self.max_triangle_prob = max_triangle_prob
        self.m_scale = m_scale

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(self.m_scale * np.log2(num_nodes))
        m = np.random.choice(max_m) + 1
        p = np.random.uniform(high=self.max_triangle_prob)
        graph = nx.powerlaw_cluster_graph(num_nodes, m, p)
        c = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(c).copy()
        logging.debug('Generated {}-node powerlaw cluster graph with max m: {}'.format(
                num_nodes, max_m))
        return graph

def get_generator(sizes, size_prob=None, dataset_len=None):
    # gen_prob = [12/49, 12/49, 12/49, 12/49, 1/49]
    gen_prob = [1/4, 1/4, 1/4, 1/4]
    max_size = max(sizes)
    generator = dataset.EnsembleGenerator(
        [
        ERGenerator(sizes, m_scale= 0.8, size_prob=size_prob),
        WSGenerator(sizes, m_scale= 1, size_prob=size_prob),
        BAGenerator(sizes, size_prob=size_prob),
        PowerLawClusterGenerator(sizes, size_prob=size_prob),
        # BAGenerator([s for s in range(max_size, 5*max_size, 5)], max_p= 0.1, m_scale= 1,size_prob=size_prob),
        PowerLawClusterGenerator([s for s in range(max_size, 5*max_size)], max_triangle_prob= 0.25, m_scale= 1, size_prob=size_prob)
        ],
        gen_prob=gen_prob,
        dataset_len=dataset_len)
    #print(generator)
    return generator

def get_dataset(task, dataset_len, sizes, size_prob=None, **kwargs):
    generator = get_generator(sizes, size_prob=size_prob,
        dataset_len=dataset_len)
    return dataset.GraphDataset(
        None, task=task, generator=generator, **kwargs)

def main():
    sizes = np.arange(6, 31)
    dataset = get_dataset("graph", sizes)
    print('On the fly generated dataset has length: {}'.format(len(dataset)))
    example_graph = dataset[0]
    print('Example graph: nodes {}; edges {}'.format(example_graph.G.nodes, example_graph.G.edges))

    print('Even the same index causes a new graph to be generated: edges {}'.format(
            dataset[0].G.edges))

    print('This generator has no label: {}, '
          '(but can be augmented via apply_transform)'.format(dataset.num_node_labels))

if __name__ == '__main__':
    main()
