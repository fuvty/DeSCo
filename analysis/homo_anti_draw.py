import numpy as np
import networkx as nx
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subgraph_counting.data import gen_query_ids

if __name__ == "__main__":
    # Load the data
    data = [
        0.19,
        0.4,
        0.39,
        0.14,
        0.52,
        0.38,
        0.24,
        0.43,
        0.31,
        0.26,
        0.67,
        0.24,
        0.55,
        0.16,
        0.82,
        0.04,
        0.83,
        0.25,
        0.47,
        0.37,
        0.28,
        0.41,
        0.69,
        0.23,
        0.49,
        0.28,
        0.31,
        0.29,
        0.60,
        0.28,
        0.84,
        0.2,
        0.70,
        0.21,
        0.50,
        0.2,
        0.73,
        0.13,
        0.74,
        0.2,
        0.89,
        0.15,
        0.81,
        0.09,
        0.93,
        0.1,
        0.75,
        0.12,
        0.86,
        0.12,
        0.84,
        0.07,
        0.95,
        0.06,
        0.97,
        0.03,
        1.00,
        0.01,
    ]

    # define the query graphs
    query_ids = gen_query_ids(query_size=[3, 4, 5])
    # query_ids = [6]
    nx_queries = [nx.graph_atlas(i) for i in query_ids]

    # define the size of nodes
    size = [len(g) for g in nx_queries]
    degree = [np.mean([d for n, d in g.degree()]) for g in nx_queries]

    # homo data is the even pos of data
    data_dict = {}
    data_dict["homophily"] = [data[i] for i in range(0, 58, 2)]
    data_dict["antisymmetry"] = [data[i] for i in range(1, 58, 2)]
    data_dict["size"] = size
    data_dict["degree"] = degree
    # data_dict['color'] = [data[i]-data[i+1] for i in range(0, 58, 2)]
    # data_dict['scalar'] = [100 for i in range(0, 58, 2)]
    df = pd.DataFrame(data_dict)
    # set xmin and xmax of sns

    # set node size
    sns.set(font_scale=1.35, style="whitegrid")
    # sns.set_theme(style="whitegrid")
    # sns.set(rc={'figure.figsize':(5,5)})
    g = sns.relplot(
        x="homophily", y="antisymmetry", data=df, s=100, size="degree", palette="tab10"
    )
    # plt.axis('equal')# set limit of x and y
    g.set(xlim=(0, 1.1), ylim=(0, 0.55))

    # seaborn save figure
    g.figure.savefig("analysis/output/homo_anti.pdf")
    g.figure.savefig("analysis/output/homo_anti.png")
