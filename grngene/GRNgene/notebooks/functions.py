import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import random
from scipy.stats import pearsonr
import GRNgene as gg
import cma
import numpy as np
import pickle as pkl
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit

# def power_law(x, a):
#     return (a-1) * np.power(x, -a)

# def power_law(x, a, b):
#     return a * np.power(x, b)

def power_law(x, b):
    return np.power(x, b)

def plot_degrees(
    G: nx.DiGraph,
    plot_fit: bool = False
):
    """
    Plot the total, in- and out- degree distribution of a network and optionally fit a power law.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the gene regulatory network.
    
    plot_fit : bool, optional (default=False)
        If True, fit and overlay a power-law curve on the degree distribution plot.
        
    Returns
    -------
    None
        Displays the plots and prints the fitted power law parameters if `plot_fit` is True.
    """
    N = G.number_of_nodes()
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)
    degree_counts = np.unique(degree_sequence, return_counts=True)
    degrees = degree_counts[0]
    counts = degree_counts[1]
    probabilities = counts / N
               
    in_degree_sequence = sorted((d for n, d in G.in_degree()), reverse=True)
    in_dmax = max(in_degree_sequence)
    in_degree_counts = np.unique(in_degree_sequence, return_counts=True)
    in_degrees = in_degree_counts[0]
    in_counts = in_degree_counts[1]
    in_probabilities = in_counts / N

    out_degree_sequence = sorted((d for n, d in G.out_degree()), reverse=True)
    out_dmax = max(out_degree_sequence)
    out_degree_counts = np.unique(out_degree_sequence, return_counts=True)
    out_degrees = out_degree_counts[0]
    out_counts = out_degree_counts[1]
    out_probabilities = out_counts / N
    
    fig = plt.figure("Degree distribution", figsize=(8, 8))
    axgrid = fig.add_gridspec(3, 2)

    # Total degree
    ax0 = fig.add_subplot(axgrid[0, 0])
    mask = degrees > 0
    ax0.scatter(np.log(degrees[mask]), np.log(probabilities[mask]), marker="o", color="b")
    ax0.set_title("log-log Degree Distribution")
    ax0.set_ylabel("log P(k)")
    ax0.set_xlabel("log Degree k")
    
    # Fit and plot power law
    params = None
    if plot_fit:
        params, _ = curve_fit(power_law, degrees[degrees > 0], probabilities[degrees > 0])
        ax0.plot(
            np.log(degrees[degrees > 0]),
            np.log(power_law(degrees[degrees > 0], *params)),
            'r--',
            label='Power law fit'
        )
        ax0.legend()
    if params is not None:
        # print(f"Fitted power law parameters: a = {params[0]:.2f}, b = {params[1]:.2f}")
        print(f"Fitted power law parameter: b = {params[0]:.2f}")
    
    # Degree histogram
    ax1 = fig.add_subplot(axgrid[0, 1])
    ax1.bar(degrees, counts)
    ax1.set_title("Degree Histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Number of Nodes")

    # In-degree
    ax2 = fig.add_subplot(axgrid[1, 0])
    in_mask = in_degrees > 0
    ax2.scatter(np.log(in_degrees[in_mask]), np.log(in_probabilities[in_mask]), marker="o", color="b")
    ax2.set_title("Log-Log In-Degree Distribution")
    ax2.set_ylabel("Log P(k)")
    ax2.set_xlabel("Log Degree k")

    # Fit and plot power law
    params = None
    if plot_fit:
        params, _ = curve_fit(power_law, in_degrees[in_degrees > 0], in_probabilities[in_degrees > 0])
        ax2.plot(
            np.log(in_degrees[in_degrees > 0]),
            np.log(power_law(in_degrees[in_degrees > 0], *params)),
            'r--',
            label='Power law fit'
        )
        ax2.legend()

    # Degree histogram
    ax2 = fig.add_subplot(axgrid[1, 1])
    ax2.bar(in_degrees, in_counts)
    ax2.set_title("Degree Histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    
    # Out-degree
    ax3 = fig.add_subplot(axgrid[2, 0])
    out_mask = out_degrees > 0
    ax3.scatter(np.log(out_degrees[out_mask]), np.log(out_probabilities[out_mask]), marker="o", color="b")
    ax3.set_title("Log-Log Out-Degree Distribution")
    ax3.set_ylabel("Log P(k)")
    ax3.set_xlabel("Log Degree k")

    # Fit and plot power law
    params = None
    if plot_fit:
        params, _ = curve_fit(power_law, out_degrees[out_degrees > 0], out_probabilities[out_degrees > 0])
        ax3.plot(
            np.log(out_degrees[out_degrees > 0]),
            np.log(power_law(out_degrees[out_degrees > 0], *params)),
            'r--',
            label='Power law fit'
        )
        ax3.legend()

    # Degree histogram
    ax4 = fig.add_subplot(axgrid[2, 1])
    ax4.bar(out_degrees, out_counts)
    ax4.set_title("Out-Degree Histogram")
    ax4.set_xlabel("Degree")
    ax4.set_ylabel("Number of Nodes")

    fig.tight_layout()
    plt.show()

    

def connect_components_by_hubs(G: nx.DiGraph, hub_bias=3.0, mode='auto', verbose=False):
    """
    Connect disconnected weakly connected components in a directed graph
    by adding edges between hubs (high-degree nodes).

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph (modified in-place).
    hub_bias : float
        Bias factor (> 1 favors high-degree nodes).
    mode : str
        'auto' (default): randomly pick direction
        'out' : source -> target
        'in'  : target -> source
    verbose : bool
        Print progress information.
    """
    W = G.to_undirected()
    components = list(nx.connected_components(W))
    if len(components) <= 1:
        if verbose:
            print("Graph is already weakly connected.")
        return G

    if verbose:
        print(f"Connecting {len(components)} components...")

    # Sort components by size
    components = sorted(components, key=len, reverse=True)

    for i in range(len(components) - 1):
        compA = components[i]
        compB = components[i + 1]

        # Degree dictionary (total degree)
        degA = {n: G.degree(n) for n in compA}
        degB = {n: G.degree(n) for n in compB}

        # Compute hub-biased probabilities
        nodesA, weightsA = zip(*[(n, (d + 1e-3) ** hub_bias) for n, d in degA.items()])
        nodesB, weightsB = zip(*[(n, (d + 1e-3) ** hub_bias) for n, d in degB.items()])
        probA = np.array(weightsA) / np.sum(weightsA)
        probB = np.array(weightsB) / np.sum(weightsB)

        u = np.random.choice(nodesA, p=probA)
        v = np.random.choice(nodesB, p=probB)

        # Determine direction
        if mode == 'auto':
            if random.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        elif mode == 'out':
            G.add_edge(u, v)
        elif mode == 'in':
            G.add_edge(v, u)
        else:
            raise ValueError("mode must be 'auto', 'in', or 'out'.")

    return G

def connect_isolated_nodes_to_hubs(G: nx.DiGraph, hub_bias=3.0, mode='out', verbose=False):
    """
    Connect isolated nodes (degree 0) to high-degree nodes using preferential attachment.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph (modified in-place).
    hub_bias : float
        Bias factor for hub selection (> 1 favors high-degree nodes).
    mode : str
        'out': isolated node -> hub
        'in' : hub -> isolated node
        'auto': random choice between out and in
    verbose : bool
        Whether to print info during processing.
    """
    degrees = dict(G.degree())
    isolated_nodes = [n for n, d in degrees.items() if d == 0]
    hub_candidates = [n for n, d in degrees.items() if d > 0]

    if not isolated_nodes:
        if verbose:
            print("No isolated nodes found.")
        return G

    if not hub_candidates:
        raise ValueError("No nodes with non-zero degree to attach to.")

    # Compute hub-biased probabilities
    weights = [(degrees[n] + 1e-3) ** hub_bias for n in hub_candidates]
    prob = np.array(weights) / sum(weights)

    for iso_node in isolated_nodes:
        hub = np.random.choice(hub_candidates, p=prob)

        if mode == 'out':
            if not G.has_edge(iso_node, hub):
                G.add_edge(iso_node, hub)
                if verbose:
                    print(f"Added edge: {iso_node} → {hub}")
        elif mode == 'in':
            if not G.has_edge(hub, iso_node):
                G.add_edge(hub, iso_node)
                if verbose:
                    print(f"Added edge: {hub} → {iso_node}")
        elif mode == 'auto':
            if random.random() < 0.5:
                if not G.has_edge(iso_node, hub):
                    G.add_edge(iso_node, hub)
                    if verbose:
                        print(f"Added edge: {iso_node} → {hub}")
            else:
                if not G.has_edge(hub, iso_node):
                    G.add_edge(hub, iso_node)
                    if verbose:
                        print(f"Added edge: {hub} → {iso_node}")
        else:
            raise ValueError("mode must be 'out', 'in', or 'auto'.")

    return G

import networkx as nx
import numpy as np
from collections import Counter
import random

def enforce_degree1(G: nx.DiGraph, verbose=False):
    """
    Modify G in-place by removing edges from nodes of the highest degree 
    that has a higher or equal frequency than degree 1, until degree 1 
    is the most frequent in the graph.
    """
    iteration = 0
    while True:
        degrees_dict = dict(G.degree())
        degree_sequence = sorted(degrees_dict.values(), reverse=True)
        degrees_arr, freqs_arr = np.unique(degree_sequence, return_counts=True)

        # Check if degree 1 is most frequent
        max_freq = freqs_arr.max()
        freq_deg1 = freqs_arr[degrees_arr.tolist().index(1)] if 1 in degrees_arr else 0

        if freq_deg1 == max_freq and (freqs_arr > freq_deg1).sum() == 0:
            if verbose:
                print(f"[{iteration}] Degree 1 is now the most frequent.")
            break

        # Find the highest degree with freq ≥ freq of degree 1
        violation_degrees = [deg for deg, freq in zip(degrees_arr, freqs_arr)
                             if freq >= freq_deg1 and deg != 1]
        if not violation_degrees:
            break

        target_degree = max(violation_degrees)
        candidates = [n for n, d in degrees_dict.items() if d == target_degree]
        if not candidates:
            break

        random.shuffle(candidates)
        for node in candidates:
            in_edges = list(G.in_edges(node))
            out_edges = list(G.out_edges(node))
            all_edges = in_edges + out_edges
            if not all_edges:
                continue
            edge = random.choice(all_edges)
            G.remove_edge(*edge)
            if verbose:
                print(f"[{iteration}] Removed edge {edge} from node {node} (deg={target_degree})")
            break  # Reassess after one removal

        iteration += 1

    return G

def remove_nan_values(species: list) -> None:
    """
    Removes NaN values from the end of the gene and interaction lists of
    a species.

        Args:
            - species (list): A list containing two lists:
            [gene_list, interaction_list]
    """
    # Check if the last element is not a string (i.e., it's NaN)
    while type(species[0][-1]) is not str:
        species[0].pop()  # Remove the last gene entry
        species[1].pop()  # Remove the corresponding interaction entry

#!/usr/bin/env python3

import networkx as nx
import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from GRNgene.GRN.genesGroup import subgraph3N

# Load the data from an Excel file
if __name__ == "__main__":
    # If running as a script, load the Excel file directly
    document = pd.read_excel("../data/41598_2021_3625_MOESM5_ESM.xlsx")
else:
    # If imported as a module, use pkg_resources to locate the file
    excel_path = pkg_resources.resource_filename(
        "ochunGRN", "GRN/41598_2021_3625_MOESM5_ESM.xlsx")
    document = pd.read_excel(excel_path)

# Load the data for different species from the Excel file
arabidopsisThaliana = (document["""Supplementary Table S1: Networks. A spreadsheet file with filtered networks"""].tolist()[2:], document["Unnamed: 1"].tolist()[2:], "Arabidopsis thaliana")  # noqa : E501
drosophilaMelanogaster = (document["Unnamed: 2"].tolist()[2:], document["Unnamed: 3"].tolist()[2:], "Drosophila Melanogaster")  # noqa : E501
escherichniaColi = (document["Unnamed: 4"].tolist()[2:], document["Unnamed: 5"].tolist()[2:], "Escherichnia coli")  # noqa : E501
homoSapiens = (document["Unnamed: 6"].tolist()[2:], document["Unnamed: 7"].tolist()[2:], "Homo sapiens")  # noqa : E501
saccharomycesCerevisiae = (document["Unnamed: 8"].tolist()[2:], document["Unnamed: 9"].tolist()[2:], "Saccharomyces cerevisiae")  # noqa : E501

def create_graph(species: list) -> nx.DiGraph:
    """
    Creates a directed graph from gene interactions for a given species.

        Args:
            - species (list): A list containing two lists:
            [gene_list, interaction_list]

        Returns:
            nx.DiGraph: A directed graph representing gene interactions.
    """
    N = len(species[0])  # Number of gene interactions
    Graph = nx.DiGraph()  # Create an empty directed graph
    for i in range(N):
        Graph.add_edge(species[0][i], species[1][i])  # Add edges to the graph
    return Graph


def find_autoregulatory_genes(species: list) -> tuple:
    """
    Finds genes that regulate themselves (autoregulatory genes) in a given
    species.

        Args:
            - species (list): A list containing two lists:
            [gene_list, interaction_list]

        Returns:
            tuple: A tuple containing the list of autoregulatory genes
            and their proportion.
    """
    autoregulatory_genes = []
    N = len(species[0])  # Number of gene interactions
    for i in range(N):
        if species[0][i] == species[1][i]:  # Check if a gene regulates itself
            autoregulatory_genes.append(species[0][i])
    # Return the genes and their proportion
    return (autoregulatory_genes, len(autoregulatory_genes) / len(species[0]))


def find_double_regulatory_genes(graph: nx.DiGraph) -> tuple:
    """
    Finds pairs of genes that regulate each other in a given species graph.

        Args:
            - graph (nx.DiGraph): A directed graph representing gene
            interactions.

        Returns:
            tuple: A tuple containing the list of double-regulatory gene pairs
            and their proportion.
    """
    double_regulation_genes = []
    for nodeA in graph:  # Iterate over all nodes (genes)
        # Iterate over successors of each node (genes regulated by nodeA)
        for nodeB in graph.successors(nodeA):
            # Check if nodeA is regulated by nodeB
            if nodeA in graph.successors(nodeB):
                double_regulation_genes.append((nodeA, nodeB))
    # Return the pairs and their proportion
    return (double_regulation_genes,
            len(double_regulation_genes) / graph.number_of_edges())


def calculate_ffl_ratio(graph: nx.DiGraph) -> None:
    """
    Calculates and prints the ratio of feed-forward loops (FFL) in a graph.

        Args:
            - graph (nx.DiGraph): A directed graph representing gene
            interactions.
    """
    print(graph.number_of_nodes(), graph.number_of_edges())
    print(nx.is_weakly_connected(graph))
    dict_ffl = subgraph3N(graph)
    motifs = {v: 0 for v in dict_ffl.values()}  # Initialize motif counts
    N = len(dict_ffl)
    print(N)
    for i in dict_ffl.keys():
        motifs[dict_ffl[i]] += 1
    # Print sorted motifs by count
    print(sorted(motifs.items(), key=lambda item: item[1]))
    print(motifs['FFL'] / N)  # Print the ratio of FFL motifs


def plot_autoregulatory_distribution(graph: nx.DiGraph,
                                     group_count: int) -> None:
    """
    Plots the distribution of autoregulatory genes based on degree groups.

    Args:
    graph (nx.DiGraph): A directed graph representing gene interactions.
    group_count (int): The number of groups to divide the nodes by degree.
    """
    autoregulatory_dict = {}
    total_dict = {}
    degree_list = list(graph.degree())
    degree_values = [degree_list[i][1] for i in range(len(degree_list))]
    degree_values.sort()
    gene_count = graph.number_of_nodes()
    # Create degree group pivot points
    pivot_points = [degree_values[int(i * gene_count / (group_count + 1))]
                    for i in range(1, group_count)]
    print(pivot_points)

    for node in graph:
        group_val = 0
        deg = graph.degree(node)
        while group_val < group_count - 1 and deg > pivot_points[group_val]:
            group_val += 1
        if node in graph.successors(node):  # If a node is autoregulatory
            if group_val in autoregulatory_dict:
                autoregulatory_dict[group_val] += 1
            else:
                autoregulatory_dict[group_val] = 1
        if group_val in total_dict:
            total_dict[group_val] += 1
        else:
            total_dict[group_val] = 1

    sorted_autoregulatory = sorted(autoregulatory_dict.items())
    print(sorted_autoregulatory)

    X, Y = [], []
    for item in sorted_autoregulatory:
        deg = item[0]
        X.append(str(item[0]))
        # Normalize by total number of nodes in each group
        Y.append(item[1] / total_dict[deg])
    plt.bar(X, Y)
    plt.show()


def find_high_degree_nodes(graph: nx.DiGraph, min_degree: int = 10) -> list:
    """
    Finds nodes (genes) in the graph with a degree greater than a specified
    minimum.

        Args:
            - graph (nx.DiGraph): A directed graph representing gene
            interactions.
            - min_degree (int): The minimum degree threshold.

        Returns:
            list: A list of nodes with a degree higher than the specified
            threshold.
    """
    return [node for node in graph if graph.degree(node) > min_degree]


def main():
    # for species in [arabidopsisThaliana,
    #                 drosophilaMelanogaster,
    #                 escherichniaColi,
    #                 homoSapiens,
    #                 saccharomycesCerevisiae]:

    for species in [escherichniaColi,
                    homoSapiens]:        
        
        remove_nan_values(species)
        graph = create_graph(species)
        print(f"\nSpecies: {species[2]}")
        print("Nodes:", graph.number_of_nodes())
        print("Edges:", graph.number_of_edges())
        print("Autoregulatory Genes:", find_autoregulatory_genes(species))
        print("Double Regulatory Genes:", find_double_regulatory_genes(graph))
        clustering_coefficients = list(nx.clustering(graph).items())
        print("Average Clustering Coefficient:",
              np.mean([clustering_coefficients[i][1]
                       for i in range(graph.number_of_nodes())]))
        print("Autoregulatory Distribution:",
              plot_autoregulatory_distribution(graph, 100))
        print("Nodes with Degree >= 10:", find_high_degree_nodes(graph))


if __name__ == "__main__":
    main()
    
def get_largest_cc(graph: nx.DiGraph):
    weakly_connected_nodeset = max(nx.weakly_connected_components(graph), key=len)
    largest_cc = graph.subgraph(weakly_connected_nodeset).copy()
    return largest_cc
