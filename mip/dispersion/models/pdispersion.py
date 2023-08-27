import numpy as np
import pyomo.environ as pyo

from models.base import DispersionModel


def maxmin_rule(model, i, j):
    return model.D <= model.d[i, j] + model.M * (1 - model.x[i]) + model.M * (1 - model.x[j])


class PDispersion(DispersionModel):

    def __init__(self, coordinates: np.ndarray, p: int):
        super().__init__(coordinates, p)

        # More parameters
        self.M = max(self.d[:, :])

        # More variables
        self.D = pyo.Var(within=pyo.NonNegativeReals)

        # More constraints
        self.maxmin_rule = pyo.Constraint(self.A, rule=maxmin_rule)

        # Objective
        self.obj = pyo.Objective(expr=self.D, sense=pyo.maximize)
