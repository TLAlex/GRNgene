#!/usr/bin/env python3

import networkx as nx
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities, modularity
import pandas as pd

# Barabasi-Albert algorithm
# Ma = np.zeros((genesNb,genesNb))


def BarabasiAlbertAlgorithm(n: int, an: int) -> nx.graph.Graph:
    """
    Generate a random graph using the Barabasi-Albert model.

    This function creates a scale-free network based on preferential
    attachment.

        Parameters:
            - n (int): The total number of nodes in the network.
            - an (int): The number of initial connected nodes in the network.

        Returns:
            - networkx.Graph: The generated graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(an))

    # Create new nodes with edges following the preferential attachment
    for new_node in range(an, n):
        sum_denominator = 2 * G.number_of_edges() + G.number_of_nodes()
        # probabilistic list creation
        s = 0
        Lprob = []
        for node in G:
            s += (G.degree[node]+1) / sum_denominator
            Lprob.append(s)
        G.add_node(new_node)

        # new edges determination
        for a in range(an):
            random = rd.random()
            final_node = 0
            while random > Lprob[final_node]:
                final_node += 1
            G.add_edge(new_node, final_node)

    # connectivity condition
    if nx.is_connected(G):
        return G
    else:
        return BarabasiAlbertAlgorithm(n, an)

def LFRAlgorithm(
    n: int,
    tau1: float = 2.3,
    tau2: float = 1.5,
    mu: float = 0.1,
    average_degree: int = 3,
    min_community: int = 5,
    max_community: int = None,
    max_retries: int = 20,
    seed: int = None,
    connect_components: bool = True,
    hub_bias: float = 3.0
) -> nx.Graph:
    """
    Generate a graph using the LFR benchmark model, ensuring it is connected
    and contains no self-loops.

    Parameters
    ----------
    n : int
        Number of nodes in the network.
    tau1 : float, optional
        Power-law exponent for the degree distribution.
    tau2 : float, optional
        Power-law exponent for the community size distribution.
    mu : float, optional
        Mixing parameter (fraction of edges that connect to other communities).
    average_degree : int, optional
        Approximate average degree of nodes.
    min_community : int, optional
        Minimum size of communities.
    max_retries : int, optional (default=10)
        Maximum number of retries if generation fails.
    seed : int, optional
        Random seed.
    connect_components : bool, optional (default=True)
        If True, connect disconnected components after graph generation.
    hub_bias : float, optional (default=3.0)
        Bias factor for selecting hubs when connecting components.

    Returns
    -------
    nx.Graph
        A connected LFR benchmark graph with no self-loops.

    Raises
    ------
    RuntimeError
        If the graph could not be generated after max_retries.
    """
    attempt = 0
    while True:
        try:
            attempt += 1
            G = nx.generators.community.LFR_benchmark_graph(
                n=n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=average_degree,
                min_community=min_community,
                max_community=max_community,
                seed=seed
            )

            # Ensure it's a Graph (not MultiGraph)
            if not isinstance(G, nx.Graph):
                G = nx.Graph(G)

            # Remove self-loops (autoregulation)
            self_loops = list(nx.selfloop_edges(G))
            if self_loops:
                G.remove_edges_from(self_loops)
                #print(f"Removed {len(self_loops)} self-loops from the graph.")

            # Optionally connect components
            if connect_components and not nx.is_connected(G):
                print("Graph is disconnected. Connecting components...")
                G = connect_components_by_degree(G, hub_bias=hub_bias)

            print(f"Successfully generated LFR graph on attempt {attempt}.")
            return G

        except nx.ExceededMaxIterations:
            print(f"Generation failed on attempt {attempt}, retrying...")

        if attempt >= max_retries:
            raise RuntimeError(f"Exceeded max retries ({max_retries}). LFR generation failed.")

def connect_components_by_degree(G, hub_bias=3.0):
    """
    Connect disconnected components in a graph by adding edges between them.

    Nodes are selected with probability proportional to (degree ** hub_bias),
    favoring high-degree (hub) nodes when connecting components.

    Parameters
    ----------
    G : nx.Graph
        An undirected NetworkX graph that may be disconnected.

    hub_bias : float, optional (default=3.0)
        Bias factor to favor hubs: higher values increase the probability
        of selecting nodes with higher degree when connecting components.

    Returns
    -------
    nx.Graph
        The same graph, now connected (modified in place).

    Notes
    -----
    - This function modifies the input graph by adding edges between components
      until the graph becomes connected.
    - Components are connected iteratively by linking the first two components
      found at each iteration.
    - Requires an undirected graph; convert to undirected if needed using
      `G.to_undirected()`.
    """
    added_edges = 0
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        compA, compB = components[0], components[1]

        # Get node degrees for both components
        degrees_A = np.array([G.degree(n) for n in compA])
        degrees_B = np.array([G.degree(n) for n in compB])

        # Amplify degrees for hub preference
        degrees_A = (degrees_A + 1e-3) ** hub_bias
        degrees_B = (degrees_B + 1e-3) ** hub_bias

        # Compute selection probabilities within each component
        prob_A = degrees_A / degrees_A.sum()
        prob_B = degrees_B / degrees_B.sum()

        # Select nodes from each component
        nodeA = np.random.choice(list(compA), p=prob_A)
        nodeB = np.random.choice(list(compB), p=prob_B)

        # Add edge to connect components
        G.add_edge(nodeA, nodeB)
        added_edges += 1
        print(f"Connected node {nodeA} (deg={G.degree(nodeA)}) with {nodeB} (deg={G.degree(nodeB)})")

    print(f"Graph is now connected (added {added_edges} edges).")
    return G


def meanClustering(G: nx.Graph) -> float:
    """
    Calculate the mean clustering coefficient of a graph.

    Parameters:
    - G (networkx.Graph): The input graph.

    Returns:
    - float: The mean clustering coefficient.
    """
    L = list(nx.clustering(G).items())
    return np.mean([L[i][1] for i in range(G.number_of_nodes())])


def createLogVerificationScaleFree(G: nx.Graph) -> tuple:
    """
    Generate data for log-log verification of a scale-free network.

    Parameters:
    - G (networkx.Graph): The input graph.

    Returns:
    - tuple: Two lists containing log(degree) and log(proportion of nodes
    with that degree).
    """
    res = ([], [])
    dicD = {}
    N = G.number_of_nodes()

    # Calculate the degree distribution
    for node in G:
        d = G.degree[node]
        if d not in dicD:
            dicD[d] = 1
        else:
            dicD[d] += 1

    # Calculate log values for scale-free verification
    for d in dicD:
        res[0].append(np.log(d))
        res[1].append(np.log(dicD[d]/N))

    return res


def adjacenteDiMatriceFromGraph(G: nx.Graph,
                                autoRG: float,
                                duoRG: float) -> tuple:
    """
    Generate a directed graph and adjacency matrix from an undirected graph.

    This function assigns directed edges to an undirected graph based on
    auto-regulation and duo-regulation rates.

    Parameters:
    - G (networkx.Graph): The undirected graph.
    - autoRG (float): The self-regulation rate.
    - duoRG (float): The duo-regulation rate.

    Returns:
    - tuple: A directed graph and its adjacency matrix.
    """
    DiG = nx.DiGraph()
    DiG.add_nodes_from(G)

    # Assign directed edges
    for edge in G.edges():
        rdNumber = rd.random()
        if rdNumber < duoRG:
            DiG.add_edges_from((edge, edge[::-1]), color='black')
        else:
            rdNumber = rd.random()
            if rdNumber < 0.5:
                DiG.add_edges_from([edge], color='blue')
            else:
                DiG.add_edges_from([edge[::-1]], color='blue')

    # Assign self-loops
    for node in G:
        rdNumber = rd.random()
        if rdNumber < autoRG:
            DiG.add_edge(node, node, color='gray')

    # Create adjacency matrix and add activations/inhibitions
    M = nx.to_numpy_array(DiG)
    addActivationInhibition(DiG, M)
    return (DiG, M)


def adjacenteDiMatriceStaredFromGraph(G: nx.Graph,
                                      autoRG: float,
                                      duoRG: float) -> tuple:
    """
    Generate a directed graph and adjacency matrix from an undirected graph
    with a specific starting point.

    This function assigns directed edges to an undirected graph based
    on auto-regulation and duo-regulation rates,
    starting from the node with the highest degree.

    Parameters:
    - G (networkx.Graph): The undirected graph.
    - autoRG (float): The self-regulation rate.
    - duoRG (float): The duo-regulation rate.

    Returns:
    - tuple: A directed graph and its adjacency matrix.
    """
    DiG = nx.DiGraph()
    DiG.add_nodes_from(G)
    degree_dict = dict(G.degree())
    motherNode = max(degree_dict, key=degree_dict.get)
    distance = nx.shortest_path_length(G, motherNode)
    cache = set()

    # Assign directed edges from the most connected node
    for nodeA in distance:
        for nodeB in G[nodeA]:
            edge = (nodeA, nodeB)
            if edge not in cache:
                cache.add(edge)
                cache.add(edge[::-1])
                rdNumber = rd.random()
                if rdNumber < duoRG:
                    DiG.add_edges_from((edge, edge[::-1]), color='black')
                else:
                    DiG.add_edges_from([edge], color='blue')
        # Assign self-loops
        rdNumber = rd.random()
        if rdNumber < autoRG:
            DiG.add_edge(nodeA, nodeA, color='gray')

    # Create adjacency matrix and add activations/inhibitions
    M = nx.to_numpy_array(DiG)
    addActivationInhibition(DiG, M)
    return (DiG, M)


def addActivationInhibition(G: nx.Graph,
                            M: np.ndarray) -> None:
    """
    Adds activation or inhibition labels to the graph edges.

    This function randomly assigns activation or inhibition
    to each edge in the graph.

    Parameters:
    - G (networkx.Graph): The directed graph.
    - M (numpy.ndarray): The adjacency matrix of the graph.
    """
    for u, v in G.edges():
        inhibitionBool = rd.random() < 0.5
        if inhibitionBool:
            M[u][v] *= -1
            G[u][v]['acInColor'] = 'r'  # Red for inhibition
        else:
            G[u][v]['acInColor'] = 'g'  # Green for activation


def addColors(G: nx.Graph,
              M: np.ndarray) -> None:
    """
    Adds colors to the graph edges based on their type.

    This function assigns colors to edges in the graph depending
    on whether they are self-loops,
    mutual connections, or one-directional.

    Parameters:
    - G (networkx.Graph): The directed graph.
    - M (numpy.ndarray): The adjacency matrix of the graph.
    """
    for u, v in G.edges():
        if v == u:
            G[u][v]['color'] = "gray"  # Self-loop
        elif (v, u) in G.edges():
            G[u][v]['color'] = "black"  # Mutual connection
        else:
            G[u][v]['color'] = "blue"  # One-directional connection

        if M[u][v] == 1:
            G[u][v]["acInColor"] = 'g'  # Green for activation
        else:
            G[u][v]["acInColor"] = 'r'  # Red for inhibition

def adj_mx_binary(adj_mx):
    """
    Convert an adjacency matrix to a binary matrix (1 for any nonzero entry, 0 otherwise).

    Parameters
    ----------
    adj_mx : np.ndarray
        The input adjacency matrix (can be weighted or unweighted).

    Returns
    -------
    np.ndarray
        A binary adjacency matrix of the same shape, where all nonzero entries are set to 1.
    """
    binary_adj_mx = adj_mx.copy()
    binary_adj_mx[binary_adj_mx != 0] = 1.0
    return binary_adj_mx

def connect_components_by_degree(G, hub_bias=3.0):
    """
    Connect disconnected components by adding edges between them, favoring hubs.

    Parameters
    ----------
    G : nx.Graph
        An undirected NetworkX graph. The graph may be disconnected.
    
    hub_bias : float, optional (default=3.0)
        Bias factor to favor connecting nodes with higher degrees.
        Nodes are selected with probability proportional to (degree ** hub_bias).

    Returns
    -------
    nx.Graph
        The same graph, now fully connected (modified in-place).

    Notes
    -----
    - The function modifies the input graph by adding edges until it becomes connected.
    - Components are connected iteratively by selecting nodes from different components
      with probability proportional to (degree ** hub_bias).
    - Requires an undirected graph; use `G.to_undirected()` if starting from a DiGraph.
    """
    added_edges = 0
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        compA, compB = components[0], components[1]  # keep original logic

        # Get the node degrees for both components
        degrees_A = np.array([G.degree(n) for n in compA])
        degrees_B = np.array([G.degree(n) for n in compB])

        # Amplify degrees for hub preference
        degrees_A = (degrees_A + 1e-3) ** hub_bias
        degrees_B = (degrees_B + 1e-3) ** hub_bias

        # Define the selection probability of nodes within each component
        prob_A = degrees_A / degrees_A.sum()
        prob_B = degrees_B / degrees_B.sum()

        # Select the node for each component
        nodeA = np.random.choice(list(compA), p=prob_A)
        nodeB = np.random.choice(list(compB), p=prob_B)

        # Connect the components with the selected nodes
        G.add_edge(nodeA, nodeB)
        added_edges += 1
        print(f"Connected node {nodeA} (deg={G.degree(nodeA)}) with {nodeB} (deg={G.degree(nodeB)})")

    print(f"Graph is now connected (added {added_edges} edges).")
    return G

def network_properties(G):
    """
    Compute properties of a directed network.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph (NetworkX DiGraph).

    Returns
    -------
    dict
        A dictionary containing:
        
        - 'avg_clustering' : float
            Average clustering coefficient of the network.
        
        - 'avg_degree' : float
            Average degree of nodes in the network.
        
        - 'degrees' : np.ndarray
            Array of unique degrees in the network.
        
        - 'density' : float
            Density of the network.
        
        - 'degree_proba' : np.ndarray
            Probability distribution of node degrees (degree frequencies normalized by number of nodes).
        
        - 'modularity_value' : float or None
            Modularity of the network (based on the undirected version). None if computation fails.
        
        - 'nb_edges' : int
            Total number of edges in the network.
        
        - 'nb_nodes' : int
            Total number of nodes in the network.
        
        - 'strongly_connected' : bool
            Whether the network is strongly connected (every node reachable from every other, considering direction).
        
        - 'weakly_connected' : bool
            Whether the network is weakly connected (every node reachable from every other, ignoring direction).
    """
    res = {}
    
    # Degree distribution
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    N = G.number_of_nodes()
    degree_counts = np.unique(degree_sequence, return_counts=True)
    degrees = degree_counts[0]
    counts = degree_counts[1]
    degree_probabilities = counts / N

    modularity_value = None  # Initialize to None in case of failure
    try:
        communities = greedy_modularity_communities(G.to_undirected())
        modularity_value = modularity(G.to_undirected(), communities)
    except Exception as e:
        print(f"Error in community detection or modularity calculation: {e}")

    res['avg_clustering'] = nx.average_clustering(G)
    res['avg_degree'] = sum(dict(G.degree()).values()) / N
    res['degrees'] = degrees
    res['density'] = nx.density(G)
    res['degree_proba'] = degree_probabilities
    res['modularity_value'] = modularity_value
    res['nb_edges'] = G.number_of_edges()
    res['nb_nodes'] = N
    res['strongly_connected'] = nx.is_strongly_connected(G)
    res['weakly_connected'] = nx.is_weakly_connected(G)

    return res

def adj_mx_gnw_goldstandard(filepath: str) -> np.ndarray:
    """
    Convert a GeneNetWeaver edge list to an adjacency matrix.

    Parameters
    ----------
    filepath : str
        Path to the edge list (.tsv file) generated by GeneNetWeaver.
        The file must have three columns: source, target, and interaction.

    Returns
    -------
    np.ndarray
        A binary adjacency matrix as a NumPy array (shape: [n_nodes, n_nodes]),
        where 1 indicates an interaction and 0 indicates no interaction.
    """
    df = pd.read_csv(filepath, sep="\t", header=None, names=["source", "target", "interaction"])
    active_edges = df[df["interaction"] == 1]
    G = nx.from_pandas_edgelist(active_edges, source="source", target="target", create_using=nx.DiGraph())
    return nx.adjacency_matrix(G).toarray()

##############################################################################


def main():
    genesNb = 5
    autoRG = 0.05
    duoRG = 0.1
    Ma = np.random.randint(2, size=(genesNb, genesNb))
    for i in [1000]:
        for j in [1, 2, 3]:
            G = BarabasiAlbertAlgorithm(i, j)
            # nx.draw(G, with_labels=True, font_weight='bold')
            # for node in G:
            #    print(node, G.degree[node])
            print(i, j)
            meanClustering(G)
            print()
            coord = createLogVerificationScaleFree(G)
            plt.scatter(coord[0], coord[1], label=f"am = {j}")
            plt.xlabel("log d")
            plt.ylabel("log(P(d)/N)")
            plt.legend()
            plt.title("Scale-free verification")
    plt.savefig("Melvin/Images/scaleFreeProof.png")
    G = BarabasiAlbertAlgorithm(20, 2)
    meanClustering(G)

    # G = nx.DiGraph()
    # G.add_nodes_from(range(genesNb))
    #
    # for i in range(genesNb):
    #    for j in range(genesNb):
    #        if Ma[i,j] :
    #            G.add_edge(i,j)
    #

    # print(list(G.edges()))

    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.subplot(122)
    DiG = adjacenteDiMatriceStaredFromGraph(G, autoRG, duoRG)[0]
    edges = DiG.edges()
    colors = [DiG[u][v]['color'] for u, v in edges]
    nx.draw_kamada_kawai(DiG, with_labels=True,
                         font_weight='bold', edge_color=colors)
    plt.show()

    K = []
    for i in range(genesNb):
        for j in range(genesNb):
            if Ma[i, j]:
                k = rd.random()
                K.append(k)
##############################################################################


if __name__ == "__main__":
    main()
