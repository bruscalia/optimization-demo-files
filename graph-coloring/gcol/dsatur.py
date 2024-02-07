from gcol.graph import Color, Graph, Node


class DSatur(Graph):
    """Graph Coloring DSatur Algorithm proposed by Brélaz (1979)

    Brélaz, D., 1979. New methods to color the vertices of a graph.
    Communications of the ACM, 22(4), 251-256.
    """

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
        Q = [node for node in self.N]  # Pool of uncolored nodes
        while len(Q) > 0:
            node: Node = min(Q, key=lambda x: (x.saturation, x.degree))
            Q.remove(node)
            next_color = self.find_next_color(node)
            node.set_color(next_color)
            if save_history:
                self.history.append(node)
        self.C.sort(key=lambda x: x.n_nodes, reverse=True)

    @property
    def cost(self):
        return len(self.C)
