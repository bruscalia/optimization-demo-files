from typing import List, Tuple

import gif
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import networkx as nx

from gcol.graph import Node


def draw_from_nodes(
    nodes: List[Node],
    **kwargs
):
    N = [n.index for n in nodes]
    C = [n.color.index for n in nodes]
    E = [(n.index, m.index) for n in nodes for m in n.neighbors]
    draw_colored_graph(N, C, E, **kwargs)



def draw_colored_graph(
    nodes: List[int],
    colors: List[int],
    edges: List[Tuple[int, int]],
    ax=None,
    plot_colors=None,
    node_size=200,
    node_alpha=1.0,
    font_size=8,
    edge_color="grey",
    edge_alpha=0.2,
    use_labels=True,
    plot_margins=True,
    layout_iter=100,
    seed=None
):

    # Create a list of colors base on two colormaps
    if plot_colors is None:
        plot_colors = get_cmap('Dark2').colors + get_cmap('Set1').colors + get_cmap('Set2').colors + get_cmap('Set3').colors

    # Create a networkx graph from the edges
    G = nx.Graph()
    G.add_edges_from(edges)

    # Map the nodes to the colors from the color_map
    node_color_list = [plot_colors[color] for color in colors]

    # Draw the graph
    pos = nx.spring_layout(G, iterations=layout_iter, seed=seed)  # positions for all nodes, you can use other layouts like shell, random, etc.
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, node_color=node_color_list,
        node_size=node_size, ax=ax, alpha=node_alpha,
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, edge_color=edge_color)
    if use_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)

    # Possibly remove margins
    if not plot_margins:
        plt.axis("off")


def draw_colored_gif(
    filename: str,
    history: List[Node],
    colors=None,
    node_size=200,
    node_alpha=1.0,
    font_size=8,
    edge_color="grey",
    edge_alpha=0.2,
    neighbor_colors="#900C3F",
    use_labels=True,
    plot_margins=True,
    layout_iter=100,
    duration=200,
    seed=None,
    **kwargs
):
    nodes: List[Node] = sorted(history, key=lambda x: x.index)

    # Obtain edges and colors
    edges = []
    node_colors = []
    node_indexes = []
    for i in nodes:
        node_colors.append(i.color.index)
        node_indexes.append(i.index)
        for j in i.neighbors:
            edges.append((i.index, j.index))

    # Create a list of colors base on two colormaps
    if colors is None:
        colors = get_cmap('Dark2').colors + get_cmap('Set1').colors + get_cmap('Set2').colors + get_cmap('Set3').colors

    # Create a networkx graph from the edges
    G = nx.Graph()
    G.add_edges_from(edges)

    # Map the nodes to the colors from the color_map
    node_color_list = [colors[color] for color in node_colors]

    # Draw the graph
    pos = nx.spring_layout(G, iterations=layout_iter, seed=seed)  # positions for all nodes, you can use other layouts like shell, random, etc.

    @gif.frame
    def new_frame(i: int):
        fig, ax = plt.subplots(**kwargs)
        nx.draw_networkx_nodes(
            G, pos, nodelist=node_indexes[i:], node_color="grey",
            node_size=node_size, ax=ax, alpha=0.5 * node_alpha,
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=node_indexes[:i], node_color=node_color_list[:i],
            node_size=node_size, ax=ax, alpha=node_alpha,
        )
        n = nodes[i - 1]
        n_edges = [(n.index, j.index) for j in n.neighbors]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, edge_color=edge_color)
        nx.draw_networkx_edges(G, pos, edgelist=n_edges, ax=ax, alpha=edge_alpha * 0.5,
                               edge_color=neighbor_colors)
        if use_labels:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)
        fig.tight_layout()
        if not plot_margins:
            plt.axis("off")

    # Construct "frames"
    frames = [new_frame(i + 1) for i in range(len(history))]

    # Save "frames" to gif with a specified duration (milliseconds) between each frame
    gif.save(frames, filename, duration=duration)
    plt.close()
