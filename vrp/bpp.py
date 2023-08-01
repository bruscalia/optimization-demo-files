from typing import Any, Dict

import numpy as np
import pyomo.environ as pyo


def capacity_constraint(model, k):
    return sum(model.x[i, k] * model.q[i] for i in model.I) <= model.Q


def demand_constraint(model, i):
    return sum(model.x[i, :]) == 1.0


def active_bin(model, i, k):
    return model.x[i, k] <= model.y[k]


def create_bpp(demands: Dict[Any, float], capacity: float):
    
    # Feasible limits
    min_bins = np.ceil(sum(demands.values()) / capacity)
    max_bins = 2 * int(min_bins)
    
    # Create model instance
    model = pyo.ConcreteModel()
    
    # Sets
    model.I = pyo.Set(initialize=demands.keys())
    model.K = pyo.Set(initialize=range(int(max_bins)))
    
    # Parameters
    model.q = pyo.Param(model.I, initialize=demands)
    model.Q = pyo.Param(initialize=capacity)
    
    # Variables
    model.x = pyo.Var(model.I, model.K, within=pyo.Binary)
    model.y = pyo.Var(model.K, within=pyo.Binary)
    
    # Constraints
    model.capacity_constraint = pyo.Constraint(model.K, rule=capacity_constraint)
    model.demand_constraint = pyo.Constraint(model.I, rule=demand_constraint)
    model.active_bin = pyo.Constraint(model.I, model.K, rule=active_bin)
    
    # Objective
    model.obj = pyo.Objective(expr=sum(model.y[:]), sense=pyo.minimize)
    
    return model
    