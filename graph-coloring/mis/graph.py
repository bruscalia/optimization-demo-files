import copy
from typing import Dict, List, Optional, Tuple


class Node:

    neighbors: List['Node']
    index: int
    selected: bool

    def __init__(self, index):
        self.index = index
        self.neighbors = []
        self.selected = False

    def __repr__(self) -> str:
        return f"N{self.index}"

    def add_neighbor(self, node: 'Node'):
        if node not in self.neighbors:
            self.neighbors.append(node)

    def delete(self):
        for n in self.neighbors:
            n.neighbors.remove(self)

    @property
    def degree(self):
        return len(self.neighbors)


class Graph:

    N: Dict[int, Node]
    history: List[Node]

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        nodes: Optional[List[int]] = None
    ):
        """Undirected Graph base class

        Parameters
        ----------
        edges : List[Tuple[int, int]]
            List of edges for which nodes can't be assigned to the same color

        nodes : Optional[List[int]], optional
            List of node indexes for which colors should be defined, by default None
        """

        # Start the set
        if nodes is None:
            self.N = {}
        else:
            self.N = {i: Node(i) for i in nodes}

        # Include all neighbors
        for i, j in edges:
            self._new_edge(i, j)

        self.history = []

    @property
    def active_nodes(self):
        return [node for node in self.N.values() if node.selected]

    @property
    def inactive_nodes(self):
        return [node for node in self.N.values() if not node.selected]

    @property
    def nodelist(self):
        return list(self.N.values())

    @property
    def queue(self):
        return [n for n in self.nodelist if not n.selected]

    @property
    def size(self):
        return len(self.N)

    def _new_node(self, i: int):
        if i not in self.N:
            self.N[i] = Node(i)

    def _new_edge(self, i: int, j: int):
        self._new_node(i)
        self._new_node(j)
        self.N[i].add_neighbor(self.N[j])
        self.N[j].add_neighbor(self.N[i])

    def select(self, node: Node):
        node.selected = True
        selected_neighbors = node.neighbors.copy()
        for n in selected_neighbors:
            other = self.N.pop(n.index)
            other.delete()
        self.history.append(node)

    def deactivate(self):
        for n in self.N.values():
            n.selected = False

    def activate(self):
        for n in self.N.values():
            n.selected = True

    def copy(self):
        """Creates deepcopy of a graph

        Returns
        -------
        Graph
            Deepcopy of instance
        """
        return copy.deepcopy(self)
