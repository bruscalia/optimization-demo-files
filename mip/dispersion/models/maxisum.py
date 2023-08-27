import numpy as np
import pyomo.environ as pyo

from models.base import DispersionModel
from models.pdispersion import PDispersion


def dispersion_c1(model, i, j):
    return model.z[i, j] <= model.x[i]


def dispersion_c2(model, i, j):
    return model.z[i, j] <= model.x[j]


def dispersion_c3(model, i, j):  # Unnecessary
    return model.z[i, j] >= model.x[i] + model.x[j] - 1


def disp_obj(model):
    return sum(model.z[i, j] * model.d[i, j] for (i, j) in model.A)


def composed_constr(model):
    return model.D >= model.d_opt



class MaxiSum(DispersionModel):

    def __init__(self, coordinates: np.ndarray, p: int):
        super().__init__(coordinates, p)

        # More variables
        self.z = pyo.Var(self.A, within=pyo.Binary)

        # More constraints
        self.dispersion_c1 = pyo.Constraint(self.A, rule=dispersion_c1)
        self.dispersion_c2 = pyo.Constraint(self.A, rule=dispersion_c2)

        # Objective
        self.obj = pyo.Objective(rule=disp_obj, sense=pyo.maximize)


class Hybrid(PDispersion):

    def solve(self, solver: pyo.SolverFactory, **kwargs):

        # Solve p-dispersion
        solver.solve(self)

        # More parameters
        self.d_opt = pyo.Param(initialize=self.obj())
        self.obj.deactivate()

        # More variables
        self.z = pyo.Var(self.A, within=pyo.Binary)

        # More constraints
        self.dispersion_c1 = pyo.Constraint(self.A, rule=dispersion_c1)
        self.dispersion_c2 = pyo.Constraint(self.A, rule=dispersion_c2)
        self.composed_constr = pyo.Constraint(rule=composed_constr)

        # Objective
        self.obj_disp = pyo.Objective(rule=disp_obj, sense=pyo.maximize)
        return solver.solve(self, **kwargs)
