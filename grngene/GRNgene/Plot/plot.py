import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from GRNgene.GRN.genesGroup import subgraph3N, get_all_motifs_count


def plotGraph(GenesDict: dict,
              actInhPlotBool: bool = False,
              saveName: str = None) -> None:
    """
    Plot the Gene Regulatory Network (GRN) graph from the provided gene
    dictionary.

    This function plots a graph using a circular layout.
    If the `actInhPlotBool` option is enabled, it also creates a separate plot
    to represent
    the activators and inhibitors in the network.

        Parameters:
            - GenesDict (dict): A dictionary containing information about
            the graph, under the key "Graph."
            - actInhPlotBool (bool, optional): If True, an additional graph
            showing activators and inhibitors is plotted. Default is False.
            - saveName (str, optional): If provided, the graph will be saved
            with this file name.

        Returns:
            - None: The graph is displayed or saved depending on the
            parameters.
    """
    # Extract the graph from the gene dictionary
    Graph = GenesDict["Graph"]

    # Create a new figure for plotting
    plt.figure()

    # Get edges and their colors from the graph
    edges = Graph.edges()
    colors = [Graph[u][v]['color'] for u, v in edges]

    # Plot the main GRN graph using a circular layout
    plt.subplot(1 + actInhPlotBool, 1, 1)
    nx.draw_circular(Graph, with_labels=True,
                     font_weight='bold', edge_color=colors)
    plt.title("GRN Graph")

    # Plot activator/inhibitor graph if specified
    if actInhPlotBool:
        # Get activator/inhibitor edge colors
        acInColors = [Graph[u][v]['acInColor'] for u, v in edges]

        # Create a new subplot for activator/inhibitor graph
        plt.subplot(2, 1, 2)
        nx.draw_circular(Graph, with_labels=True,
                         font_weight='bold', edge_color=acInColors,
                         connectionstyle="arc3,rad=0.05")
        plt.title("GRN Graph Activator/Inhibitor")

    # Save the graph to a file if saveName is provided
    if saveName is not None:
        plt.savefig(saveName)


def plotSim(GenesDict: dict,
            ODEs: list = None,
            saveName: str = None) -> None:
    """
    Plot the simulation results of the ODEs for the gene regulatory network.

    Parameters
    ----------
    GenesDict : dict
        A dictionary containing the simulation results for each type of ODE.
    
    ODEs : list, optional
        A list of ODE types to plot. If not provided, it defaults to the ODEs
        in `GenesDict["ODEs"]`.
    
    saveName : str, optional
        If provided, saves the graph with this file name.

    Returns
    -------
    None
        Displays the plot or saves it if saveName is provided.
    """
    font = {'family': 'serif', 'color': 'darkred', 'size': 8}

    if ODEs is None:
        ODEs = GenesDict["ODEs"]
    if isinstance(ODEs, str):
        ODEs = [ODEs]

    genesNb = GenesDict["genesNb"]

    fig, axes = plt.subplots(len(ODEs), 1, figsize=(8, 4 * len(ODEs)))

    if len(ODEs) == 1:
        axes = [axes]  # Ensure axes is iterable

    # Plot each ODE type in its subplot
    for i, ode_type in enumerate(ODEs):
        ax = axes[i]
        for solGenes in range(genesNb):
            ax.plot(GenesDict[f"{ode_type}X"],
                    GenesDict[f"{ode_type}Y"][solGenes], label=f"Gene {solGenes}")
        ax.set_xlabel("time (h)", fontdict=font)
        ax.set_ylabel("mRNA concentrations", fontdict=font)
        ax.set_title(f"{ode_type} law simulation")

    # Create legend outside the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Adjust spacing to make room for the legend and between subplots
    plt.subplots_adjust(right=0.8, hspace=0.4)

    if saveName is not None:
        plt.savefig(saveName, bbox_inches='tight')
    plt.show()


def plotProt(GenesDict, saveName=None):
    """
    Plot the protein concentrations over time from
    the "indirect" ODE simulation.

    This function plots the simulated protein concentrations using the results
    from the "indirect" ODE model in `GenesDict`.

        Parameters:
            - GenesDict (dict): A dictionary containing the simulation results,
            including the "indirect" ODE results.
            - saveName (str, optional): If provided, the graph will be saved
            with this file name.

        Raises:
            - KeyError: If "indirect" is not in `GenesDict["ODEs"]`.

        Returns:
            - None: The graph is displayed or saved depending
            on the parameters.
    """
    # Ensure that "indirect" ODE results are available
    if "indirect" not in GenesDict["ODEs"]:
        raise

    # Get the number of genes from the gene dictionnary
    genesNb = GenesDict["genesNb"]

    # Set the font for plot labels
    font = {'family': 'serif', 'color': 'darkred', 'size': 8}

    # Plot protein concentrations for each gene
    for solGenes in range(genesNb):
        plt.plot(GenesDict["indirectX"],
                 GenesDict["indirectProt"][solGenes], label=solGenes)

        # Set plot labels and title
        plt.xlabel("time (h)", fontdict=font)
        plt.ylabel("protein concentrations", fontdict=font)
        plt.title("protein concentrations from indirect law simulation")

    # Create a legend with gene labels
    labels = []
    for solGenes in range(genesNb):
        labels.append(f"Gene {solGenes}")
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper right')

    # Save the plot to a file if saveName is provided
    if saveName is not None:
        plt.savefig(saveName)

def power_law(x, a, b):
    return a * np.power(x, b)

def plot_grn_degree(
    G: nx.DiGraph,
    plot_network: bool = False,
    plot_fit: bool = False,
    layout: str = "spring"
):
    """
    Plot the degree distribution of a gene regulatory network (GRN) and optionally fit a power law.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the gene regulatory network.
    
    plot_network : bool, optional (default=False)
        If True, also plot the network structure (can be slow for large networks).
    
    plot_fit : bool, optional (default=False)
        If True, fit and overlay a power-law curve on the degree distribution plot.
    
    layout : str, optional (default="spring")
        Layout algorithm for visualizing the network. Should match a NetworkX layout name
        (e.g., 'spring', 'kamada_kawai', 'circular', 'shell', 'spectral').
        See: https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout

    Returns
    -------
    None
        Displays the plots and prints the fitted power law parameters if `plot_fit` is True.
    """
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)
    N = G.number_of_nodes()

    fig = plt.figure("Degree distribution", figsize=(8, 8))
    axgrid = fig.add_gridspec(5, 4)

    # Plot the network structure
    if plot_network:
        ax0 = fig.add_subplot(axgrid[0:3, :])
        try:
            layout_func = getattr(nx, f"{layout}_layout")
        except AttributeError:
            available_layouts = [fn.replace('_layout', '') for fn in dir(nx) if fn.endswith('_layout')]
            raise ValueError(
                f"Unknown layout '{layout}'. Must be a valid NetworkX layout name. "
                f"Available layouts include: {', '.join(available_layouts)}"
            )
        pos = layout_func(G)
        nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Gene Regulatory Network")
        ax0.set_axis_off()

    # Degree distribution data
    degree_counts = np.unique(degree_sequence, return_counts=True)
    degrees = degree_counts[0]
    counts = degree_counts[1]
    probabilities = counts / N

    # Log-log scale plot for scale-free verification
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.scatter(np.log(degrees), np.log(probabilities), marker="o", color="b")
    ax1.set_title("Log-Log Degree Distribution")
    ax1.set_ylabel("Log P(k)")
    ax1.set_xlabel("Log Degree k")

    # Fit and plot power law
    params = None
    if plot_fit:
        params, _ = curve_fit(power_law, degrees[degrees > 0], probabilities[degrees > 0])
        ax1.plot(
            np.log(degrees[degrees > 0]),
            np.log(power_law(degrees[degrees > 0], *params)),
            'r--',
            label='Power law fit'
        )
        ax1.legend()

    # Degree histogram
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(degrees, counts)
    ax2.set_title("Degree Histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")

    fig.tight_layout()
    plt.show()

    if params is not None:
        print(f"Fitted power law parameters: a = {params[0]:.2f}, b = {params[1]:.2f}")

def plot_motifs_count(motifs_count, log=True):
    """
    Plot a bar chart of motif counts.

    Parameters
    ----------
    motifs_count : dict
        Dictionary mapping motif names (str) to their counts (int).
    
    log : bool, optional (default=True)
        If True, plot the logarithm of counts (log(count)); if False, plot raw counts.

    Returns
    -------
    None
        Displays the bar plot of motif counts.
    """
    motifs = list(motifs_count.keys())
    counts = list(motifs_count.values())

    plt.figure(figsize=(12, 6))
    if log:
        counts_log = [np.log(count) if count > 0 else 0 for count in counts]
        plt.bar(motifs, counts_log, color='skyblue')
        ylabel = 'Log Count'
    else:
        plt.bar(motifs, counts, color='skyblue')
        ylabel = 'Count'

    plt.xlabel('Motif')
    plt.ylabel(ylabel)
    plt.title('Motif Count Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compare_motif_dist(network1, network2, label1, label2, save_filename=None):
    """
    Compare motif count distributions between two networks.

    Parameters
    ----------
    network1 : nx.Graph or nx.DiGraph
        The first network to analyze.
    
    network2 : nx.Graph or nx.DiGraph
        The second network to analyze.
    
    label1 : str
        Label for the first network (used in the plot legend).
    
    label2 : str
        Label for the second network (used in the plot legend).
    
    save_filename : str, optional (default=None)
        If provided, the figure is saved to this path in EPS format.

    Returns
    -------
    None
        Displays the comparison bar plot and saves the figure if requested.

    Notes
    -----
    - The function computes motif counts using 3-node subgraphs.
    - The counts are plotted in log10 scale (log10(count + 1)) to handle zero counts.
    """
    network1_motifs = subgraph3N(network1)
    network1_motifs_count = get_all_motifs_count(network1_motifs)

    network2_motifs = subgraph3N(network2)
    network2_motifs_count = get_all_motifs_count(network2_motifs)

    # Ensure both have the same motif keys
    all_motifs = set(network2_motifs_count.keys()).union(network1_motifs_count.keys())
    motifs = sorted(all_motifs)

    counts_network1 = [network1_motifs_count.get(m, 0) for m in motifs]
    counts_network2 = [network2_motifs_count.get(m, 0) for m in motifs]

    # Convert counts to log-scale (add +1 to avoid log(0))
    log_counts_network1 = [np.log10(c + 1) for c in counts_network1]
    log_counts_network2 = [np.log10(c + 1) for c in counts_network2]

    x = np.arange(len(motifs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, log_counts_network1, width, label=label1, color='blue')
    ax.bar(x + width/2, log_counts_network2, width, label=label2, color='red')

    ax.set_ylabel('Log10(Motif Count + 1)')
    ax.set_xlabel('Motif')
    ax.set_title('Motif Count Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(motifs, rotation=45)
    ax.legend()

    plt.tight_layout()
    if save_filename is not None:
        fig.savefig(save_filename)
    plt.show()

def plot_degree_network(
    G1,
    G2,
    label1="Network 1",
    label2="Network 2",
    plot_fit=False,
    save_path=None
):
    """
    Compare the log-log degree distributions of two GRNs using a stacked layout.

    Parameters
    ----------
    G1 : nx.Graph or nx.DiGraph
        The first network to analyze and plot.
    
    G2 : nx.Graph or nx.DiGraph
        The second network to analyze and plot.
    
    label1 : str, optional (default="Network 1")
        Label for the first network (used in plot and legend).
    
    label2 : str, optional (default="Network 2")
        Label for the second network (used in plot and legend).
    
    plot_fit : bool, optional (default=False)
        If True, fits and overlays a power-law curve on the degree distribution plots.
    
    save_path : str, optional (default=None)
        If provided, saves the figure to this path. The format is inferred from the file extension.

    Returns
    -------
    None
        Displays the plot and saves it if requested.

    Notes
    -----
    - The left panel shows the two network structures (top and bottom).
    - The right panel compares their log-log degree distributions.
    """
    # Degree sequences
    degree_seq1 = sorted([d for n, d in G1.degree()], reverse=True)
    degree_seq2 = sorted([d for n, d in G2.degree()], reverse=True)

    degrees1, counts1 = np.unique(degree_seq1, return_counts=True)
    degrees2, counts2 = np.unique(degree_seq2, return_counts=True)

    prob1 = counts1 / G1.number_of_nodes()
    prob2 = counts2 / G2.number_of_nodes()

    # Set up figure: 2 rows, 2 columns (stacked left side)
    fig = plt.figure(figsize=(14, 8))
    axgrid = fig.add_gridspec(2, 3, width_ratios=[1, 0.05, 2])  # narrow left, wide right

    # Plot network 1 (top left)
    ax0 = fig.add_subplot(axgrid[0, 0])
    pos1 = nx.kamada_kawai_layout(G1)
    nx.draw_networkx_nodes(G1, pos1, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G1, pos1, ax=ax0, alpha=0.3)
    ax0.set_title(label1)
    ax0.axis('off')

    # Plot network 2 (bottom left)
    ax1 = fig.add_subplot(axgrid[1, 0])
    pos2 = nx.kamada_kawai_layout(G2)
    nx.draw_networkx_nodes(G2, pos2, ax=ax1, node_size=10)
    nx.draw_networkx_edges(G2, pos2, ax=ax1, alpha=0.3)
    ax1.set_title(label2)
    ax1.axis('off')

    # Log-log degree distribution (right side, spans both rows)
    ax2 = fig.add_subplot(axgrid[:, 2])
    ax2.scatter(np.log(degrees1 + 1e-6), np.log(prob1 + 1e-6),
                label=f"{label1}", color='blue', alpha=0.7)
    ax2.scatter(np.log(degrees2 + 1e-6), np.log(prob2 + 1e-6),
                label=f"{label2}", color='red', alpha=0.7)
    ax2.set_xlabel("Log Degree k")
    ax2.set_ylabel("Log Frequency P(k)")
    ax2.set_title("Log-Log Degree Distribution Comparison")

    if plot_fit:
        # Fit network 1
        try:
            params1, _ = curve_fit(power_law, degrees1[degrees1 > 0], prob1[degrees1 > 0])
            ax2.plot(np.log(degrees1[degrees1 > 0]),
                     np.log(power_law(degrees1[degrees1 > 0], *params1)),
                     'b--', label=f'{label1} fit (b={params1[1]:.2f})')
        except Exception as e:
            print(f"Fit failed for {label1}: {e}")

        # Fit network 2
        try:
            params2, _ = curve_fit(power_law, degrees2[degrees2 > 0], prob2[degrees2 > 0])
            ax2.plot(np.log(degrees2[degrees2 > 0]),
                     np.log(power_law(degrees2[degrees2 > 0], *params2)),
                     'r--', label=f'{label2} fit (b={params2[1]:.2f})')
        except Exception as e:
            print(f"Fit failed for {label2}: {e}")

    ax2.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    plt.show()

def compare_degree_distribution(
    G1,
    G2,
    label1="Network 1",
    label2="Network 2",
    plot_fit=False,
    save_path=None
):
    """
    Compare the log-log degree distributions of two GRNs.

    Parameters
    ----------
    G1 : nx.Graph or nx.DiGraph
        The first network to analyze.
    
    G2 : nx.Graph or nx.DiGraph
        The second network to analyze.
    
    label1 : str, optional (default="Network 1")
        Label for the first network (used in plot and legend).
    
    label2 : str, optional (default="Network 2")
        Label for the second network (used in plot and legend).
    
    plot_fit : bool, optional (default=False)
        If True, fits and overlays a power-law curve on the degree distribution plots.
    
    save_path : str, optional (default=None)
        If provided, saves the figure to this path. The format is inferred from the file extension.

    Returns
    -------
    None
        Displays the plot and saves it if requested.

    Notes
    -----
    - The degree distributions are shown in log-log scale with small epsilon (1e-6) added to avoid log(0).
    - The power-law fit uses the `power_law` function defined at the module level.
    """
    # Degree sequences
    degree_seq1 = sorted([d for n, d in G1.degree()], reverse=True)
    degree_seq2 = sorted([d for n, d in G2.degree()], reverse=True)

    degrees1, counts1 = np.unique(degree_seq1, return_counts=True)
    degrees2, counts2 = np.unique(degree_seq2, return_counts=True)

    prob1 = counts1 / G1.number_of_nodes()
    prob2 = counts2 / G2.number_of_nodes()

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(np.log(degrees1 + 1e-6), np.log(prob1 + 1e-6),
               label=f"{label1}", color='blue', alpha=0.7)
    ax.scatter(np.log(degrees2 + 1e-6), np.log(prob2 + 1e-6),
               label=f"{label2}", color='red', alpha=0.7)
    ax.set_xlabel("Log Degree k")
    ax.set_ylabel("Log Frequency P(k)")
    ax.set_title("Log-Log Degree Distribution Comparison")

    if plot_fit:
        # Fit network 1
        try:
            params1, _ = curve_fit(power_law, degrees1[degrees1 > 0], prob1[degrees1 > 0])
            ax.plot(
                np.log(degrees1[degrees1 > 0]),
                np.log(power_law(degrees1[degrees1 > 0], *params1)),
                'b--',
                label=f'{label1} fit (b={params1[1]:.2f})'
            )
        except Exception as e:
            print(f"Fit failed for {label1}: {e}")

        # Fit network 2
        try:
            params2, _ = curve_fit(power_law, degrees2[degrees2 > 0], prob2[degrees2 > 0])
            ax.plot(
                np.log(degrees2[degrees2 > 0]),
                np.log(power_law(degrees2[degrees2 > 0], *params2)),
                'r--',
                label=f'{label2} fit (b={params2[1]:.2f})'
            )
        except Exception as e:
            print(f"Fit failed for {label2}: {e}")

    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
