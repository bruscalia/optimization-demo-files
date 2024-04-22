class Item:

    index: int
    weight: float
    value: float
    density: float
    selected: bool

    def __init__(self, index, weight, value) -> None:
        self.index = index
        self.weight = weight
        self.value = value
        self.density = value / weight
        self.selected = False

    @classmethod
    def from_dict(cls, x: dict):
        index = x["index"]
        weight = x["weight"]
        value = x["value"]
        return cls(index, weight, value)