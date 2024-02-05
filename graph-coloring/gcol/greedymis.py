from gcol.graph import Color, Graph


class GreedyMIS(Graph):
    """Greedy heuristic for maximum independent set problem
    """
    color = Color(0)

    def solve(self, save_history=False):
        """Solve the instance

        Parameters
        ----------
        save_history : bool, optional
            Either or not to store a sequence of colores nodes in the `history` attribute,
            by default False
        """
        Q = [n for n in self.N]  # Pool of all nodes
        while len(Q) > 0:

            # Sort by number of active neighbors (should be removed from possible solution)
            Q.sort(key=lambda x: (x.active_neighbors), reverse=False)
            node = Q.pop(0)
            node.set_color(self.color)
            if save_history:
                self.history.append(node)

            # Remove all neighbors from Q
            for neighbor in node.neighbors:
                if neighbor in Q:
                    Q.remove(neighbor)

    @property
    def cost(self):
        return sum(1 for n in self.N if n.active)
