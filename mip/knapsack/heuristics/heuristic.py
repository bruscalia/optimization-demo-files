from item import Item
from typing import Dict, List, Union


class BaseConstructive:

    items: List[Item]
    capacity: float
    solution: List[Item]

    def __init__(self, capacity: float, items: List[Dict[str, Union[int, float]]]) -> None:
        self.items = []
        self.capacity = capacity
        for new_element in items:
            item = Item.from_dict(new_element)
            self.items.append(item)
        self.solution = []

    @property
    def cost(self):
        return sum(i.value for i in self.solution)

    def solve(self):
        remaining = self.capacity
        for item in self.items:
            if remaining >= item.weight:
                item.selected = True
                self.solution.append(item)
                remaining = remaining - item.weight


class GreedyConstructive(BaseConstructive):

    def solve(self):
        self.items.sort(key=lambda x: x.density, reverse=True)
        super().solve()
