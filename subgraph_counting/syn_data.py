from networkx import Graph
from networkx.utils import py_random_state
from typing import Callable, Optional
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import pandas as pd
import random

# temp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import sqrt

DELTA = 1e-3


def gen_RandomTree(node: int) -> Graph:
    """
    Generate a random tree with node number as expected value
    """
    return nx.random_tree(node)


def random_connect_components(graph: Graph, components: list[set]) -> Graph:
    """
    Randomly connect all the components in the graph by adding one edge between two components
    """

    component_num = len(components)
    component_tree = gen_RandomTree(component_num)

    for edge in component_tree.edges:
        node1 = np.random.choice(list(components[edge[0]]))
        node2 = np.random.choice(list(components[edge[1]]))
        graph.add_edge(node1, node2)

    return graph


def gen_ERGraph(node: int, edge: int, connected: bool = True) -> Graph:
    """
    Generate an undirected ER graph with node number and edge number as expected value
    """
    # E(e) = p * n * (n - 1) / 2

    p = 2 * edge / (node * (node - 1))
    graph = nx.erdos_renyi_graph(node, p, directed=False)

    if connected:
        # connect the components if the graph is not connected
        components = [cc for cc in nx.connected_components(graph)]
        if len(components) > 1:
            graph = random_connect_components(graph, components)

    return graph


def gen_WSGraph(node: int, edge: int, connected: bool = True, p: float = 0.1) -> Graph:
    """
    Generate an undirected WS graph with node number and edge number as exact value;
    ALERT: usually the edge number a multiply (e.g. 1x, 2x) of node number
    """
    k = int(2 * edge / node)
    k = min(k, node - 1)

    if connected:
        try:
            graph = nx.connected_watts_strogatz_graph(node, k, p)
        except nx.exception.NetworkXPointlessConcept:
            # due to implementation problem, the networkx function may raise an exception sometime
            # graph = nx.watts_strogatz_graph(node, k, p)
            graph = nx.gnm_random_graph(node, edge, directed=False)
            # connect the components if the graph is not connected
            components = [cc for cc in nx.connected_components(graph)]
            if len(components) > 1:
                graph = random_connect_components(graph, components)
    else:
        graph = nx.watts_strogatz_graph(node, k, p)

    assert len(graph) > 0

    return graph


def gen_RandomGraph(node: int, edge: int, connected: bool = True) -> Graph:
    """
    Generate an undirected uniformly chosen random graph with node number and edge number as exact value
    """
    graph = nx.gnm_random_graph(node, edge, directed=False)

    if connected:
        # connect the components if the graph is not connected
        components = [cc for cc in nx.connected_components(graph)]
        if len(components) > 1:
            graph = random_connect_components(graph, components)
        else:
            assert nx.is_connected(graph)

    return graph


def gen_BAGraph(node: int, edge: int, connected: bool = True) -> Graph:
    """
    Generate a barabasi albert graph graph with node number and edge number as exact value
    ALERT: usually the edge number a multiply (e.g. 1x, 2x) of node number
    """
    m = int(edge / node)

    # make m larger than 1 and smaller than node - 1
    m = max(1, m)
    m = min(node - 1, m)

    graph = nx.barabasi_albert_graph(node, m)

    if connected:
        # connect the components if the graph is not connected
        components = [cc for cc in nx.connected_components(graph)]
        if len(components) > 1:
            graph = random_connect_components(graph, components)
        else:
            assert nx.is_connected(graph)

    return graph


def gen_EBAGraph(
    node: int, edge: int, connected: bool = True, p: float = 0.1, q: float = 0.1
) -> Graph:
    """
    Generate a extended barabasi albert graph graph with node number and edge number as exact value
    """
    # E(e) = m * n + p * m
    m = int(edge / node)

    # make m larger than 1 and smaller than node - 1
    m = max(1, m)
    m = min(node - 1, m)

    # adjust p to make E(e) = edge
    p = (edge - m * node) / node

    # make p+q smaller than 1 by adjusting p and q
    if p + q >= 1:
        # Warning("p + q > 1, adjust p and q")
        s = p + q
        p = p / (s) - DELTA
        q = q / (s) - DELTA

    graph = extended_barabasi_albert_graph(node, m, p, q)

    if connected:
        # connect the components if the graph is not connected
        components = [cc for cc in nx.connected_components(graph)]
        if len(components) > 1:
            graph = random_connect_components(graph, components)
        else:
            assert nx.is_connected(graph)

    return graph


def gen_PowerGraph(
    node: int, edge: int, connected: bool = True, p: float = 0.1
) -> Graph:
    """
    Generate a power law graph graph with node number and edge number as exact value
    """
    # E(e) = m * (n - m) + p * (n - m) * (m - 1)
    # m = int(edge/(node*(1 + p)))
    if node**2 - 4 * edge > 0:
        m = int((node - sqrt(node**2 - 4 * edge)) / 2)
        # p = edge/((node-m)*m)-1
        if m > 1:
            p = (edge - (node - m) * m) / ((m - 1) * (node - m))
        else:
            p = 0.0  # if m = 1, p = 0 is the only solution

        while p < 0:
            m -= 1
            p = edge / ((node - m) * m) - 1
    else:
        m = int(node / 2)
        p = 0.0

    p = min(p, 1)

    graph = powerlaw_cluster_graph(node, m, p)

    if connected:
        # connect the components if the graph is not connected
        components = [cc for cc in nx.connected_components(graph)]
        if len(components) > 1:
            graph = random_connect_components(graph, components)
        else:
            assert nx.is_connected(graph)

    return graph


@py_random_state(4)
def extended_barabasi_albert_graph(n, m, p, q, seed=None):
    """Returns an extended Barabási–Albert model graph.

    An extended Barabási–Albert model graph is a random graph constructed
    using preferential attachment. The extended model allows new edges,
    rewired edges or new nodes. Based on the probabilities $p$ and $q$
    with $p + q < 1$, the growing behavior of the graph is determined as:

    1) With $p$ probability, $m$ new edges are added to the graph,
    starting from randomly chosen existing nodes and attached preferentially at the other end.

    2) With $q$ probability, $m$ existing edges are rewired
    by randomly choosing an edge and rewiring one end to a preferentially chosen node.

    3) ALWAYS $m$ new nodes are added to the graph
    with edges attached preferentially.

    When $p = q = 0$, the model behaves just like the Barabási–Alber model.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges with which a new node attaches to existing nodes
    p : float
        Probability value for adding an edge between existing nodes. p + q < 1
    q : float
        Probability value of rewiring of existing edges. p + q < 1
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n`` or ``1 >= p + q``

    References
    ----------
    .. [1] Albert, R., & Barabási, A. L. (2000)
       Topology of evolving networks: local events and universality
       Physical review letters, 85(24), 5234.
    """
    if m < 1 or m >= n:
        msg = f"Extended Barabasi-Albert network needs m>=1 and m<n, m={m}, n={n}"
        raise nx.NetworkXError(msg)
    if p + q >= 1:
        msg = f"Extended Barabasi-Albert network needs p + q <= 1, p={p}, q={q}"
        raise nx.NetworkXError(msg)

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)

    # List of nodes to represent the preferential attachment random selection.
    # At the creation of the graph, all nodes are added to the list
    # so that even nodes that are not connected have a chance to get selected,
    # for rewiring and adding of edges.
    # With each new edge, nodes at the ends of the edge are added to the list.
    attachment_preference = []
    attachment_preference.extend(range(m))

    # Start adding the other n-m nodes. The first node is m.
    new_node = m
    while new_node < n:
        a_probability = seed.random()

        # Total number of edges of a Clique of all the nodes
        clique_degree = len(G) - 1
        clique_size = (len(G) * clique_degree) / 2

        # Adding m new edges, if there is room to add them
        if a_probability < p and G.size() <= clique_size - m:
            # Select the nodes where an edge can be added
            elligible_nodes = [nd for nd, deg in G.degree() if deg < clique_degree]
            for i in range(m):
                # Choosing a random source node from elligible_nodes
                src_node = seed.choice(elligible_nodes)

                # Picking a possible node that is not 'src_node' or
                # neighbor with 'src_node', with preferential attachment
                prohibited_nodes = list(G[src_node])
                prohibited_nodes.append(src_node)
                # This will raise an exception if the sequence is empty
                dest_node = seed.choice(
                    [nd for nd in attachment_preference if nd not in prohibited_nodes]
                )
                # Adding the new edge
                G.add_edge(src_node, dest_node)

                # Appending both nodes to add to their preferential attachment
                attachment_preference.append(src_node)
                attachment_preference.append(dest_node)

                # Adjusting the elligible nodes. Degree may be saturated.
                if G.degree(src_node) == clique_degree:
                    elligible_nodes.remove(src_node)
                if (
                    G.degree(dest_node) == clique_degree
                    and dest_node in elligible_nodes
                ):
                    elligible_nodes.remove(dest_node)

        # Rewiring m edges, if there are enough edges
        elif p <= a_probability < (p + q) and m <= G.size() < clique_size:
            # Selecting nodes that have at least 1 edge but that are not
            # fully connected to ALL other nodes (center of star).
            # These nodes are the pivot nodes of the edges to rewire
            elligible_nodes = [nd for nd, deg in G.degree() if 0 < deg < clique_degree]
            for i in range(m):
                # Choosing a random source node
                node = seed.choice(elligible_nodes)

                # The available nodes do have a neighbor at least.
                neighbor_nodes = list(G[node])

                # Choosing the other end that will get dettached
                src_node = seed.choice(neighbor_nodes)

                # Picking a target node that is not 'node' or
                # neighbor with 'node', with preferential attachment
                neighbor_nodes.append(node)
                dest_node = seed.choice(
                    [nd for nd in attachment_preference if nd not in neighbor_nodes]
                )
                # Rewire
                G.remove_edge(node, src_node)
                G.add_edge(node, dest_node)

                # Adjusting the preferential attachment list
                attachment_preference.remove(src_node)
                attachment_preference.append(dest_node)

                # Adjusting the elligible nodes.
                # nodes may be saturated or isolated.
                if G.degree(src_node) == 0 and src_node in elligible_nodes:
                    elligible_nodes.remove(src_node)
                if dest_node in elligible_nodes:
                    if G.degree(dest_node) == clique_degree:
                        elligible_nodes.remove(dest_node)
                else:
                    if G.degree(dest_node) == 1:
                        elligible_nodes.append(dest_node)

        # Adding new node with m edges
        # else:
        # Select the edges' nodes by preferential attachment
        targets = _random_subset(attachment_preference, m, seed)
        G.add_edges_from(zip([new_node] * m, targets))

        # Add one node to the list for each new edge just created.
        attachment_preference.extend(targets)
        # The new node has m edges to it, plus itself: m + 1
        attachment_preference.extend([new_node] * (m + 1))
        new_node += 1
    return G


@py_random_state(3)
def powerlaw_cluster_graph(n, m, p, seed=None):
    """Holme and Kim algorithm for growing graphs with powerlaw
    degree distribution and approximate average clustering.

    Parameters
    ----------
    n : int
        the number of nodes
    m : int
        the number of random edges to add for each new node
    p : float,
        Probability of adding a triangle after adding a random edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The average clustering has a hard time getting above a certain
    cutoff that depends on `m`.  This cutoff is often quite low.  The
    transitivity (fraction of triangles to possible triangles) seems to
    decrease with network size.

    It is essentially the Barabási–Albert (BA) growth model with an
    extra step that each random edge is followed by a chance of
    making an edge to one of its neighbors too (and thus a triangle).

    This algorithm improves on BA in the sense that it enables a
    higher average clustering to be attained if desired.

    It seems possible to have a disconnected graph with this algorithm
    since the initial `m` nodes may not be all linked to a new node
    on the first iteration like the BA model.

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m <= n`` or `p` does not
        satisfy ``0 <= p <= 1``.

    References
    ----------
    .. [1] P. Holme and B. J. Kim,
       "Growing scale-free networks with tunable clustering",
       Phys. Rev. E, 65, 026107, 2002.
    """

    if m < 1 or n < m:
        raise nx.NetworkXError(f"NetworkXError must have m>1 and m<n, m={m},n={n}")

    if p > 1 or p < 0:
        raise nx.NetworkXError(f"NetworkXError p must be in [0,1], p={p}")

    G = nx.empty_graph(m)  # add m initial nodes (m0 in barabasi-speak)
    repeated_nodes = list(G.nodes())  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = m  # next node is m
    while source < n:  # Now add the other n-1 nodes
        possible_targets = _random_subset(repeated_nodes, m, seed)
        # do one preferential attachment for new node
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)  # add one node to list for each new link
        count = 1
        while count < m:  # add m-1 more new links
            if seed.random() < p:  # clustering step: add triangle
                neighborhood = [
                    nbr
                    for nbr in G.neighbors(target)
                    if not G.has_edge(source, nbr) and not nbr == source
                ]
                if neighborhood:  # if there is a neighbor without a link
                    nbr = seed.choice(neighborhood)
                    G.add_edge(source, nbr)  # add triangle
                    repeated_nodes.append(nbr)
                    # count = count + 1 # ALERT: do not count the triangle link as a new link
                    continue  # go to top of while loop
            # else do preferential attachment step if above fails
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = count + 1

        repeated_nodes.extend([source] * m)  # add source node to list m times
        source += 1
    return G


def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


class BetaDistributionSampler:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self):
        return np.random.beta(self.a, self.b)


class SyntheticGraphGenerator:
    """
    generate a combined synthetic graph
    """

    def __init__(
        self,
        generator_names: list[str],
        generator_probs: list[float],
        node_num_sampler: Callable,
        edge_num_sampler: Callable,
        connected: bool = True,
        post_process: Optional[Callable] = None,
        use_loggers: bool = False,
    ):

        self.node_num_sampler = node_num_sampler
        self.edge_num_sampler = edge_num_sampler
        self.connected = connected
        self.post_process = post_process
        self.use_loggers = use_loggers

        # normalize the probabilities
        self.generator_probs: np.ndarray = np.array(generator_probs) / np.sum(
            generator_probs
        )

        # get the generator functions based on the names
        self.generators = []
        self.generator_names = []
        for name in generator_names:
            if name == "ER":
                self.generators.append(gen_ERGraph)
            elif name == "WS":
                self.generators.append(lambda n, m, c: gen_WSGraph(n, m, c, 0.1))
            elif name == "Random":
                self.generators.append(gen_RandomGraph)
            elif name == "BA":
                self.generators.append(gen_BAGraph)
            elif name == "EBA":
                self.generators.append(lambda n, m, c: gen_EBAGraph(n, m, c, 0.1, 0.1))
            elif name == "Power":
                self.generators.append(gen_PowerGraph)
            else:
                raise ValueError(f"unknown generator name {name}")
            self.generator_names.append(name)

        # loggers
        if use_loggers:
            self.logger = {"node_num": [], "edge_num": [], "generator_name": []}

        # store the generated graphs
        self.graphs = []

    def generate_dataset(self, num_graphs: int) -> list[nx.Graph]:
        """
        generate a list of graphs
        """
        for i in tqdm(range(num_graphs)):
            graph = self._generate_graph(i)
            self.graphs.append(graph)
        return self.graphs

    def _generate_graph(self, sid: int = 0) -> nx.Graph:
        """
        generate one graph
        """
        # sample the number of nodes and edges
        node_num = self.node_num_sampler(sid=sid)
        edge_num = self.edge_num_sampler(node_num, sid=sid)

        # sample the generator
        generator, name = self._sample_generator(node_num, edge_num)

        # generate the graph
        graph = generator(node_num, edge_num, self.connected)

        # post process the graph
        if self.post_process is not None:
            graph = self.post_process(graph)

        # log the graph
        if self.use_loggers:
            self.logger["node_num"].append(node_num)
            self.logger["edge_num"].append(edge_num)
            self.logger["generator_name"].append(name)

        return graph

    def _sample_generator(self, node_num: int, edge_num: int) -> Callable:
        """
        sample the generator based on the probabilities
        """
        # sample the generator
        generator_idx = np.random.choice(
            len(self.generator_probs), p=self.generator_probs
        )
        generator = self.generators[generator_idx]
        name = self.generator_names[generator_idx]

        return generator, name


def gen_Synthetic(
    num_graph: int, min_size: int, max_size: int
) -> tuple[list[Graph], dict]:
    if num_graph == 1827:
        print("generating 1827 synthetic graphs with standard settings...")
        graphs, _ = _gen_Synthetic_1827(num_graph, min_size, max_size)
    else:
        graphs, _ = _gen_Synthetic(num_graph, min_size, max_size)
    return graphs


def _gen_Synthetic(
    num_graph: int, min_size: int, max_size: int
) -> tuple[list[Graph], dict]:
    """
    generate a combined synthetic graph
    """
    generator_names = [
        "ER",
        "WS",
        "Random",
        "BA",
        "EBA",
        "Power",
    ]

    generator_probs = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    node_num_sampler = lambda: np.random.randint(
        min_size, max_size
    )  # uniform distribution for the number of nodes

    def avg_degree_sampler(node_num: int):
        return np.random.uniform(1, 5)

    def edge_num_sampler(node_num: int):
        avg_degree = avg_degree_sampler(node_num)
        avg_edge_num = int(node_num * avg_degree)
        # edge_num = int(np.random.f(50, 50) * avg_edge_num) # TODO: when the node number is large, make the variance smaller
        edge_num = int(
            np.random.normal(1, 0.25 - 0.125 / max_size * node_num) * avg_edge_num
        )  # when the node number is large, make the variance smaller

        # make sure the edge number is not too large nor too small for a connected graph
        edge_num = min(edge_num, node_num * (node_num - 1) // 2)
        edge_num = max(edge_num, node_num - 1)

        return edge_num

    generator = SyntheticGraphGenerator(
        generator_names,
        generator_probs,
        node_num_sampler,
        edge_num_sampler,
        connected=True,
        post_process=lambda g: nx.convert_node_labels_to_integers(
            g, first_label=0, ordering="increasing degree"
        ),
        use_loggers=True,
    )

    graphs = generator.generate_dataset(num_graph)

    return graphs, generator.logger


def _gen_Synthetic_1827(
    num_graph: int, min_size: int, max_size: int
) -> tuple[list[Graph], dict]:
    """
    generate 1827 combined synthetic graph
    """
    generator_names = [
        "ER",
        "WS",
        "Random",
        "BA",
        "EBA",
        "Power",
    ]

    generator_probs = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    def node_num_sampler(sid: int) -> int:
        # return np.random.randint(min_size, max_size)  # uniform distribution for the number of nodes

        # first: sample 60 * 23 nodes uniformly in 10:1: 59
        if sid < 60 * 23:
            node = sid // 23 + 10
        else:
            # second: sample 149 * 3 nodes uniformly in 60:5:800
            node = 5 * ((sid - 1380) // 3) + 60
            distortion = np.random.triangular(-5, 0, 5)
            node = int(distortion + node)

        return node

    def avg_degree_sampler(node_num: int, sid: int):
        # return np.random.uniform(1, 5)

        # first: sample 60 * 23 nodes uniformly in 1:0.5:12
        if sid < 60 * 23:
            return 0.5 * (sid % 23) + 1 + np.random.triangular(-0.5, 0, 0.5)
        else:
            # second: sample 149 * 3 nodes uniformly in 1:1:3
            degree = 1 * (sid - 1380) % 3 + 1
            if degree == 1:
                degree += np.random.triangular(0, 0, 1)
            elif degree == 2:
                degree += np.random.triangular(-1, 0, 1)
            elif degree == 3:
                degree += np.random.triangular(-1, 0, 0)
            else:
                raise ValueError("degree should be 1, 2 or 3")
            return degree

    def edge_num_sampler(node_num: int, sid: int):
        avg_degree = avg_degree_sampler(node_num, sid)
        avg_edge_num = int(node_num * avg_degree)
        # edge_num = int(np.random.f(50, 50) * avg_edge_num)
        edge_num = int(
            np.random.normal(1, 0.1) * avg_edge_num
        )  # when the node number is large, make the variance smaller

        # make sure the edge number is not too large nor too small for a connected graph
        edge_num = min(edge_num, node_num * (node_num - 1) // 2)
        edge_num = max(edge_num, node_num - 1)

        return edge_num

    def random_relabel_nodes(g: nx.Graph):
        """
        relabel nodes randomly
        """
        node_list = list(g.nodes)
        random.shuffle(node_list)
        mapping = {node_list[i]: i for i in range(len(node_list))}
        g = nx.relabel_nodes(g, mapping)
        return g

    generator = SyntheticGraphGenerator(
        generator_names,
        generator_probs,
        node_num_sampler,
        edge_num_sampler,
        connected=True,
        # post_process=lambda g: nx.convert_node_labels_to_integers(
        #     g, first_label=0, ordering="increasing degree"
        # ),
        post_process=random_relabel_nodes,
        use_loggers=True,
    )

    graphs = generator.generate_dataset(num_graph)

    return graphs, generator.logger


def test_generators():
    """
    deprecated
    """
    df_dict = {"node_num": [], "edge_num": [], "generator_name": [], "nx_graph": []}

    avg_degree = 2.3
    num_graphs = 10

    generator_names = [
        "ER",
        "WS",
        "Random",
        "BA",
        "EBA",
        "Power",
    ]

    generators = [
        gen_ERGraph,
        lambda n, m, c: gen_WSGraph(n, m, c, 0.1),
        gen_RandomGraph,
        gen_BAGraph,
        lambda n, m, c: gen_EBAGraph(n, m, c, 0.1, 0.1),
        gen_PowerGraph,
    ]

    for name, generator in zip(generator_names, generators):
        for num_node in range(10, 1000):
            num_edge = int(num_node * avg_degree)
            for i in range(num_graphs):
                g = generator(num_node, num_edge, True)
                df_dict["node_num"].append(g.number_of_nodes())
                df_dict["edge_num"].append(g.number_of_edges())
                df_dict["generator_name"].append(name)
                df_dict["nx_graph"].append(g)

    df = pd.DataFrame(df_dict)
    print(df)

    for name in generator_names:
        print(name)
        print(df[df["generator_name"] == name].describe())
        print()

    draw_graphs(df)


def draw_graphs(df):
    ############################ draw the graph ############################
    output_dir = "analysis/output/syn"

    filename = "syn-node-edge-reg"
    plt.figure(figsize=(16, 10))

    sns.lmplot(
        x="node_num",
        y="edge_num",
        hue="generator_name",
        data=df,
    )

    # save the figure in the output directory, if "tsne.png" already exists
    # then append a number to the filename
    i = 0
    while True:
        full_name = os.path.join(output_dir, filename + "_" + str(i) + ".png")
        if not os.path.exists(full_name):
            break
        i += 1
    print("saving figure to", os.path.abspath(full_name))
    plt.savefig(full_name, bbox_inches="tight")

    filename = "syn-node-degree-reg"
    plt.figure(figsize=(16, 10))

    df["degree"] = df["edge_num"] / df["node_num"]

    sns.lmplot(
        x="node_num",
        y="degree",
        hue="generator_name",
        data=df,
    )

    # save the figure in the output directory, if "tsne.png" already exists
    # then append a number to the filename
    i = 0
    while True:
        full_name = os.path.join(output_dir, filename + "_" + str(i) + ".png")
        if not os.path.exists(full_name):
            break
        i += 1
    print("saving figure to", os.path.abspath(full_name))
    plt.savefig(full_name, bbox_inches="tight")


def analyze_graphs(df):
    ############################ analyze the graph ############################
    graphs = df["nx_graph"].values

    clustering = [nx.average_clustering(G) for G in graphs]
    path_length = [nx.average_shortest_path_length(G) for G in graphs]
    diameter = [nx.diameter(G) for G in graphs]
    density = [nx.density(G) for G in graphs]
    avg_degree = [np.mean([d for n, d in G.degree()]) for G in graphs]

    # add to df
    df["clustering"] = clustering
    df["path_length"] = path_length
    df["diameter"] = diameter
    df["density"] = density
    df["avg_degree"] = avg_degree

    ############################ draw the graph ############################
    filename = "comprehensive"
    plt.figure(figsize=(16, 10))

    sns.pairplot(
        hue="generator_name",
        data=df,
    )

    # save the figure in the output directory, if "tsne.png" already exists
    # then append a number to the filename
    i = 0
    while True:
        full_name = os.path.join(output_dir, filename + "_" + str(i) + ".png")
        if not os.path.exists(full_name):
            break
        i += 1
    plt.savefig(full_name, bbox_inches="tight", format="pdf")
    plt.savefig(full_name, bbox_inches="tight", format="png")


if __name__ == "__main__":
    dataset, statistics = _gen_Synthetic_1827(1827, 10, 800)

    df = pd.DataFrame(statistics)
    df["nx_graph"] = dataset

    print(df)

    draw_graphs(df)

    # analyze_graphs(df)
