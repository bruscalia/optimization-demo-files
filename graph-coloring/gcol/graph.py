from typing import List, Tuple


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
    def active(self):
        return self.color is not None

    @property
    def neighbor_colors(self):
        return [n.color for n in self.neighbors if n.active]

    @property
    def saturation(self):
        return len(set((n.color for n in self.neighbors if n.active)))

    @property
    def active_neighbors(self):
        return sum(1 for n in self.neighbors if n.active)

    @property
    def degree(self):
        return len(self.neighbors)


class Graph:

    N: List[Node]
    C: List[Color]
    history: List[Node]

    def __init__(self, nodes: List[int], edges: List[Tuple[int, int]]):
        """Undirected Graph base class

        Parameters
        ----------
        nodes : List[int]
            List of node indexes for which colors should be defined

        edges : List[Tuple[int, int]]
            List of edges for which nodes can't be assigned to the same color
        """
        N = [Node(i) for i in nodes]
        for e in edges:
            i, j = e
            N[i].add_neighbor(N[j])
            N[j].add_neighbor(N[i])
        self.N = N
        self.C = []
        self.history = []
