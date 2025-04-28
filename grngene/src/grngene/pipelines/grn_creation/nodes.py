"""
This is a boilerplate pipeline 'grn_creation'
generated using Kedro 0.19.10
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
import pandas as pd

def create_custom_barabasi_albert_graph(parameters: dict[str, Any]) -> nx.Graph:
    """
    Generate a random graph using the Barabasi-Albert model.

    This function creates a scale-free network based on preferential
    attachment.

        Parameters:
            - parameters: Parameters defined in parameters_data_science.yml.
                - parameters["n"] (int): The total number of nodes in the network.
                - parameters["an"] (int): The number of initial connected nodes in the network.
                - parameters["random_seed"]: Random seed for reproducibility

        Returns:
            - networkx.Graph: The generated graph.
    """
    n = parameters.get("n", 5)
    an = parameters.get("an", 2)
    seed = parameters["random_seed"]
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(range(an))

    # Create new nodes with edges following the preferential attachment
    for new_node in range(an, n):
        sum_denominator = 2 * G.number_of_edges() + G.number_of_nodes()
        # probabilistic list creation
        s = 0
        Lprob = []
        for node in G:
            s += (G.degree[node] + 1) / sum_denominator
            Lprob.append(s)
        G.add_node(new_node)

        # new edges determination
        for a in range(an):
            random = rng.random()
            final_node = 0
            while random > Lprob[final_node]:
                final_node += 1
            G.add_edge(new_node, final_node)

    # connectivity condition
    if nx.is_connected(G):
        return G
    else:
        return create_custom_barabasi_albert_graph(parameters)


def add_edge_direction(G: nx.Graph, parameters: dict[str, Any]) -> nx.DiGraph:
    """
    Generate a directed graph and adjacency matrix from an undirected graph.

    This function assigns directed edges to an undirected graph based on
    auto-regulation and duo-regulation rates.

    Parameters:
    - G (networkx.Graph): The undirected graph.
    - parameters: Parameters defined in parameters_data_science.yml.
        - parameters['autoRG'] (float): The self-regulation rate.
        - parameters['duoRG'] (float): The duo-regulation rate.
        - parameters["random_seed"]: Random seed for reproducibility

    Returns:
    - directed_graph (networkx.DiGraph): The directed graph with edge type.
    """

    autoRG = parameters.get("autoRG", 0.2)
    duoRG = parameters.get("duoRG", 0.3)
    seed = parameters["random_seed"]
    rng = np.random.default_rng(seed)

    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(G)

    # Assign directed edges
    for edge in G.edges():
        random_number = rng.random()
        if random_number < duoRG:
            directed_graph.add_edges_from((edge, edge[::-1]))
        else:
            random_number = rng.random()
            if random_number < 0.5:
                directed_graph.add_edges_from([edge])
            else:
                directed_graph.add_edges_from([edge[::-1]])

    # Assign self-loops
    for node in G:
        random_number = rng.random()
        if random_number < autoRG:
            directed_graph.add_edge(node, node)
    return directed_graph


def add_edge_influence_type(directed_graph: nx.DiGraph, parameters: dict[str, Any]) -> nx.DiGraph:
    """
    Adds activation or inhibition labels to the graph edges.

    This function randomly assigns activation or inhibition
    to each edge in the graph.

    Parameters:
    - directed_graph (networkx.DiGraph): The directed graph.
    - parameters: Parameters defined in parameters_data_science.yml.
        - parameters["random_seed"]: Random seed for reproducibility

    Returns:
    - directed_graph (networkx.DiGraph): The directed graph with edge type.
    """
    seed = parameters["random_seed"]
    rng = np.random.default_rng(seed)

    for u, v in directed_graph.edges():
        if rng.random() < 0.5:
            directed_graph[u][v]["edge_influence_color"] = "r"
            directed_graph[u][v]["edge_influence"] = "inhibition"
        else:
            directed_graph[u][v]["edge_influence_color"] = "g"
            directed_graph[u][v]["edge_influence"] = "activation"

    return directed_graph

def nxgraph_to_df(G: nx.DiGraph) -> pd.DataFrame:
    return nx.to_pandas_edgelist(G)