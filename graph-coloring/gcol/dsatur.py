from typing import List, Tuple

from gcol.graph import Color, Node


class DSatur:

    N: List[Node]
    C: List[Color]
    history: List[Node]

    def __init__(self, nodes: List[int], edges: List[Tuple[int, int]]):
        """Graph Coloring DSatur Algorithm proposed by Brélaz (1979)

        Brélaz, D., 1979. New methods to color the vertices of a graph.
        Communications of the ACM, 22(4), 251-256.

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

    def find_next_color(self, node: Node) -> Color:
        """Finds the next available color to assign to a given node

        Parameters
        ----------
        node : Node
            Node considered in step

        Returns
        -------
        Color
            Next color available
        """
        next_color = None
        for c in self.C:
            if c not in node.neighbor_colors:
                next_color = c
                break
        if next_color is None:
            next_color = Color(len(self.C) + 1)
            self.C.append(next_color)
        return next_color

    def solve(self, save_history=False):
        """Solve the instance

        Parameters
        ----------
        save_history : bool, optional
            Either or not to store a sequence of colores nodes in the `history` attribute,
            by default False
        """
        Q = [n for n in self.N]  # Pool of uncolored nodes
        while len(Q) > 0:
            Q.sort(key=lambda x: (x.saturation, x.degree), reverse=True)
            n: Node = Q.pop(0)
            next_color = self.find_next_color(n)
            n.set_color(next_color)
            if save_history:
                self.history.append(n)
        self.C.sort(key=lambda x: x.n_nodes, reverse=True)

    @property
    def cost(self):
        return len(self.C)
