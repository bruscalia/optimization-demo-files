from typing import Dict, List, Tuple

from gcol.datamodels.colornode import Color, Node


class Graph:
    nodes: Dict[int, 'Node']
    colors: List['Color']

    def __init__(
        self,
        edges: List[Tuple[int]],
    ):
        self.nodes = {}
        for i, j in edges:
            n_i = self.get_node(i)
            n_j = self.get_node(j)
            n_i.add_neighbor(n_j)
        self.colors = []

    def get_node(self, i: int) -> Node:
        if i in self.nodes:
            return self.nodes[i]
        self.nodes[i] = Node(i)
        return self.nodes[i]

    def add_color(self):
        color = Color(len(self.colors))
        self.colors.append(color)
        return color

    @property
    def size(self):
        return len(self.nodes)

    def clean(self):
        for c in self.colors:
            c.clean_nodes()
        self.colors = []
