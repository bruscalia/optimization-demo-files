from typing import List


class Color:

    index: int
    n_nodes: int

    def __init__(self, index) -> None:
        self.index = index
        self.n_nodes = 0

    def __repr__(self):
        return f"C{self.index}"

    def add_node(self):
        self.n_nodes = self.n_nodes + 1


class Node:

    neighbors: List['Node']
    index: int
    color: Color

    def __init__(self, index):
        self.index = index
        self.neighbors = []
        self.color = None

    def __repr__(self) -> str:
        return f"N{self.index}|{self.color}"

    def add_neighbor(self, node: 'Node'):
        if node not in self.neighbors:
            self.neighbors.append(node)

    def set_color(self, color: Color):
        self.color = color
        color.add_node()

    @property
    def neighbor_colors(self):
        return [n.color for n in self.neighbors if n.color is not None]

    @property
    def saturation(self):
        return len(set((n.color for n in self.neighbors if n.color is not None)))

    @property
    def degree(self):
        return len(self.neighbors)
