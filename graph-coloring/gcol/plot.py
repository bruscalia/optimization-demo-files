from typing import Dict, List, Tuple

import gif
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colormaps

from gcol.datamodels.colornode import Node


def draw_from_nodes(
    nodes: Dict[int, Node],
    **kwargs
):
    N = [n.index for n in nodes.values()]
    N.sort(reverse=False)
    C = [nodes[n].color.index for n in N]
    E = [(n.index, m.index) for n in nodes.values() for m in n.neighbors]
    E.sort(reverse=False)
    draw_colored_graph(N, C, E, **kwargs)


def draw_colored_graph(  # noqa: PLR0913, PLR0917
    nodes: List[int],
    colors: List[int],
    edges: List[Tuple[int, int]],
    ax=None,
    plot_colors=None,
    node_size=200,
    node_alpha=1.0,
    font_size=8,
    edge_color='grey',
    edge_alpha=0.2,
    use_labels=True,
    plot_margins=True,
    layout_iter=100,
    seed=None,
):
    # Create a list of colors base on two colormaps
    if plot_colors is None:
        plot_colors = (
            colormaps['Dark2'].colors
            + colormaps['Set1'].colors
            + colormaps['Set2'].colors
            + colormaps['Set3'].colors
        )

    # Expand plot_colors to handle any number of distinct colors
    if len(colors) >= 1:
        while len(plot_colors) < max(colors) + 1:
            plot_colors += plot_colors  # Duplicate colors list

    # Create a networkx graph from the edges
    G = nx.Graph()
    G.add_edges_from(edges)

    # Map the nodes to the colors from the color_map
    node_color_list = [plot_colors[color] for color in colors]

    # Draw the graph
    pos = nx.spring_layout(
        G, iterations=layout_iter, seed=seed
    )  # positions for all nodes, you can use other layouts like shell,
    # random, etc.
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=node_color_list,
        node_size=node_size,
        ax=ax,
        alpha=node_alpha,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=edge_alpha, edge_color=edge_color
    )
    if use_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)

    # Possibly remove margins
    if not plot_margins:
        plt.axis('off')
        plt.tight_layout()

    return G, pos, ax


def draw_colored_gif(  # noqa: PLR0913, PLR0917
    filename: str,
    nodes: Dict[int, Node],
    history: List[Node],
    uncolored_color='grey',
    node_size=200,
    node_alpha=1.0,
    edge_color='grey',
    edge_alpha=0.2,
    duration=200,
    **kwargs,
):
    N = [n.index for n in nodes.values()]
    C = {n.index: n.color.index for n in nodes.values()}
    E = [(n.index, m.index) for n in nodes.values() for m in n.neighbors]
    N.sort(reverse=False)
    E.sort(reverse=False)

    @gif.frame
    def new_frame(i: int):
        Ni = [n for n in N if nodes[n] in history[:i]]
        Nx = [n for n in N if nodes[n] in history[i:]]
        Ci = [C[i] for i in Ni]
        node = history[i - 1]
        Ei = [(node.index, j.index) for j in node.neighbors]
        G, pos, ax = draw_colored_graph(
            Ni,
            Ci,
            E,
            node_alpha=node_alpha,
            node_size=node_size,
            edge_color=edge_color,
            edge_alpha=edge_alpha,
            **kwargs,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=Nx,
            node_color=uncolored_color,
            node_size=node_size,
            ax=ax,
            alpha=0.5 * node_alpha,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=Ei,
            ax=ax,
            alpha=edge_alpha * 0.5,
            edge_color=edge_color,
        )

    # Construct "frames"
    frames = [new_frame(i + 1) for i in range(len(history))]

    # Save "frames" to gif with a specified duration (milliseconds)
    # between each frame
    gif.save(frames, filename, duration=duration)
    plt.close()
