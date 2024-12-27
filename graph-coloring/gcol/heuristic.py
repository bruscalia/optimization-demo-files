import random
from typing import List, Optional

from gcol.datamodels.colornode import Color, Node
from gcol.datamodels.graph import Graph


class DSatur:
    """
    Graph Coloring DSatur Algorithm proposed by Brélaz (1979)

    Brélaz, D., 1979. New methods to color the vertices of a graph.
    Communications of the ACM, 22(4), 251-256.
    """

    graph: Optional[Graph]
    history: List['Node']

    def __init__(self):
        self.graph = None
        self.history = []
        self.queue = []

    def find_next_color(self, node: Node) -> Color:
        neighbor_colors = node.neighbor_colors
        for c in self.graph.colors:
            if c not in neighbor_colors:
                return c
        return self.graph.add_color()

    def solve(self, graph: Graph, save_history=False):
        self.graph = graph
        self.graph.clean()
        self.history = []
        self.queue = list(self.graph.nodes.values())  # Pool of uncolored nodes
        while len(self.queue) > 0:
            node = self.dequeue()
            color = self.find_next_color(node)
            node.set_color(color)
            if save_history:
                self.history.append(node)
        self.graph.colors.sort(key=lambda x: len(x.nodes), reverse=True)

    def dequeue(self):
        node = max(self.queue, key=lambda x: (x.saturation, x.degree))
        self.queue.remove(node)
        return node

    @property
    def cost(self):
        return len(self.graph.colors)


class RandomColoring(DSatur):

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def dequeue(self):
        node = self.rng.choice(self.queue)
        self.queue.remove(node)
        return node
