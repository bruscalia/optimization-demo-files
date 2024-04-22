import random
from abc import ABC, abstractmethod
from mis.graph import Graph, Node
from typing import List, Optional, Tuple


class BaseConstructive(ABC):

    graph: Graph

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        nodes: Optional[List[int]] = None,
    ):
        self.graph = Graph(edges, nodes)

    def __call__(self, *args, **kwargs):
        S = self.solve(*args, **kwargs)
        for i, n in S.N.items():
            self.graph.N[i].selected = n.selected
        self.graph.history = S.history

    @property
    def cost(self):
        return len(self.graph.active_nodes)

    @property
    def N(self):
        return self.graph.N

    @property
    def nodelist(self):
        return self.graph.nodelist

    @property
    def history(self):
        return self.graph.history

    def solve(self, *args, **kwargs) -> Graph:
        G = self.graph.copy()
        for i in range(len(G.N)):
            n = self.choice(G)
            G.select(n)
            if len(G.queue) == 0:
                assert len(G.N) == i + 1, "Unexpected behavior in remaining nodes and iterations"
                break

        return G

    @abstractmethod
    def choice(self, graph: Graph) -> Node:
        pass


class RandomChoice(BaseConstructive):

    rng: random.Random

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        nodes: Optional[List[int]] = None,
        seed=None
    ):
        super().__init__(edges, nodes)
        self.rng = random.Random(seed)

    def choice(self, graph: Graph) -> Node:
        return self.rng.choice(graph.queue)


class GreedyChoice(BaseConstructive):

    def choice(self, graph: Graph) -> Node:
        return min([n for n in graph.queue], key=lambda x: x.degree)


class MultiRandom(RandomChoice):

    def solve(self, n_iter: int = 10) -> Graph:
        best_sol = None
        best_cost = 0
        for _ in range(n_iter):
            G = super().solve()
            if len(G.N) > best_cost:
                best_cost = len(G.N)
                best_sol = G
        return best_sol
