from typing import Optional, Set


class Color:
    index: int
    nodes: Set['Node']

    def __init__(self, index, nodes=None) -> None:
        self.index = index
        if nodes is None:
            nodes = set()
        self.nodes = nodes

    def __repr__(self):
        return f'C({self.index})'

    @property
    def size(self):
        return len(self.nodes)

    def add_node(self, node: 'Node'):
        self.nodes.add(node)

    def clean_nodes(self):
        for n in self.nodes:
            n.color = None


class Node:
    index: int
    color: 'Color'
    neighbors: Set['Node']
    degree = int

    def __init__(
        self,
        index: int,
        neighbors: Optional[Set['Node']] = None,
        color: Optional['Color'] = None,
    ):
        self.index = index
        if neighbors is None:
            neighbors = set()
        self.neighbors = neighbors
        self.degree = len(self.neighbors)
        self.color = color

    def __repr__(self) -> str:
        return f'N({self.index})|{self.color}'

    @property
    def active(self):
        return self.color is not None

    @property
    def neighbor_colors(self):
        return {n.color for n in self.neighbors if n.active}

    @property
    def saturation(self):
        return len(self.neighbor_colors)

    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors.add(neighbor)
        neighbor.neighbors.add(self)
        self._update_degree()
        neighbor._update_degree()

    def remove_neighbor(self, neighbor: 'Node'):
        if neighbor in self.neighbors:
            self.neighbors.remove(neighbor)
        if self in neighbor.neighbors:
            neighbor.neighbors.remove(self)
        self._update_degree()
        neighbor._update_degree()

    def _update_degree(self):
        self.degree = len(self.neighbors)

    def set_color(self, color: Color):
        self.color = color
        color.add_node(self)
