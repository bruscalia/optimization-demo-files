from gcol.graph import Graph, Node


class GreedyMIS(Graph):
    """Greedy heuristic for maximum independent set problem
    """

    @property
    def cost(self):
        return sum(1 for n in self.N if n.active)

    def solve(self, local_search=True, save_history=False):
        """Solve the instance

        Parameters
        ----------
        local_search : bool, optional
            Either or not local search with 1-2 moves should be performed

        save_history : bool, optional
            Either or not to store a sequence of colores nodes in the `history` attribute,
            by default False
        """
        Q = [n for n in self.N]  # Pool of all nodes
        while len(Q) > 0:

            # Sort by number of active neighbors (should be removed from possible solution)
            node: Node = min(Q, key=lambda x: (x.active_neighbors))
            Q.remove(node)
            node.set_color(self.color)
            if save_history:
                self.history.append(node)

            # Remove all neighbors from Q
            for neighbor in node.neighbors:
                if neighbor in Q:
                    Q.remove(neighbor)

        if local_search:
            self.local_search()

    def local_search(self):
        """Do local search based on 1-2 swaps
        """
        proceed = True
        while proceed:
            proceed = False
            for node in self.N:
                if node.active:
                    if self._check_swaps(node):
                        proceed = True
                        continue

    def _check_swaps(self, node: Node):
        if not node.active:
            return False
        for i in node.neighbors:
            for j in node.neighbors:
                if (j is not i) and (i.active_neighbors == 1) and (j.active_neighbors == 1) and (j not in i.neighbors):
                    j.set_color(self.color)
                    i.set_color(self.color)
                    node.color = None
                    return True
        return False
