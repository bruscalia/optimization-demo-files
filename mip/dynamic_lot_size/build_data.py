import numpy as np
import pandas as pd


months = np.arange(12, dtype=int) + 1
setup_cost = np.array([85, 102, 102, 101, 98, 114, 105, 86, 119, 110, 98, 114])
demand = np.array([69, 29, 36, 61, 61, 26, 34, 67, 45, 67, 79, 56])
inventory = np.ones(12)

dataset = pd.DataFrame(
    {"setup_cost": setup_cost, "inventory_cost": inventory, "demand": demand},
    index=months,
)

dataset.to_csv("mip/dynamic_lot_size/data/input_wagner.csv")