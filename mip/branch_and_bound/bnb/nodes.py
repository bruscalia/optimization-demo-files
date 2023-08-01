from typing import List, Any, Tuple

import numpy as np
from scipy.optimize import linprog, OptimizeResult


class BaseNode:
    
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    children: List[Any]
    sol: OptimizeResult
    int_tol: float
    fathom: bool
    feasible: bool
    
    def __init__(self, branching_rule: str = "min"):
        self.branching_rule = branching_rule
        if branching_rule == "min":
            self._branching_rule = self._minimum_violation
        elif branching_rule == "max":
            self._branching_rule = self._maximum_violation
        elif branching_rule == "frac":
            self._branching_rule = self._most_fractional
        else:
            raise ValueError(f"branching_rule must be 'min', 'max', or 'frac', not {branching_rule}")
    
    @property
    def feasible(self):
        return self.sol.status <= 1
    
    @property
    def integer(self):
        if self.feasible:
            residuals = self._get_residuals()
            return np.all(residuals <= self.int_tol)
        else:
            return False
    
    def solve(self):
        bounds = list(zip(self.lb, self.ub))
        self.sol = linprog(
            self.c, A_ub=self.A_ub, b_ub=self.b_ub,
            A_eq=self.A_eq, b_eq=self.b_eq, bounds=bounds
        )
    
    def find_branch_var(self) -> Tuple[int, int, int]:
        if self.feasible:
            residuals = self._get_residuals()
            i = self._branching_rule(residuals)
            x_i = self.sol.x[i]
            floor_i = np.floor(x_i)
            ceil_i = np.ceil(x_i)
            return i, floor_i, ceil_i
        else:
            self.fathom = True
    
    def _get_residuals(self):
        x_round = np.round(self.sol.x, 0)
        return abs(self.sol.x - x_round)
    
    def _minimum_violation(self, residuals: np.ndarray):
        mask = (residuals > self.int_tol) & self.integrality
        I = np.arange(residuals.shape[0])
        j = np.argmin(residuals[mask])
        i = I[mask][j]
        return i
    
    def _maximum_violation(self, residuals: np.ndarray):
        mask = (residuals < 1 - self.int_tol) & self.integrality
        I = np.arange(residuals.shape[0])
        j = np.argmax(residuals[mask])
        i = I[mask][j]
        return i
    
    def _most_fractional(self, residuals: np.ndarray):
        mask = (residuals > self.int_tol) & self.integrality
        I = np.arange(residuals.shape[0])
        j = np.argmin(abs(residuals[mask] - 0.5))
        i = I[mask][j]
        return i
    
    def branch(self):
        i, floor_i, ceil_i = self.find_branch_var()
        xi_lb = self.lb[i]
        xi_ub = self.ub[i]
        if self.branching_rule == "max":
            self.children = [
                Node(self, i, ceil_i, xi_ub, branching_rule=self.branching_rule),
                Node(self, i, xi_lb, floor_i, branching_rule=self.branching_rule),
            ]
        else:
            self.children = [
                Node(self, i, xi_lb, floor_i, branching_rule=self.branching_rule),
                Node(self, i, ceil_i, xi_ub, branching_rule=self.branching_rule),
            ]


class Node(BaseNode):
    
    best_bound: float
    
    def __init__(self, parent: BaseNode, i: int, xi_lb: int, xi_ub: int, branching_rule="frac") -> None:
        super().__init__(branching_rule=branching_rule)
        self.parent = parent
        self.c = parent.c
        self.A_ub = parent.A_ub
        self.b_ub = parent.b_ub
        self.A_eq = parent.A_eq
        self.b_eq = parent.b_eq
        self.integrality = parent.integrality
        self.int_tol = parent.int_tol
        self.i = i
        self.xi_lb = xi_lb
        self.xi_ub = xi_ub
    
    @property
    def best_bound(self):
        return self.parent.sol.fun
    
    @property
    def lb(self):
        lb = self.parent.lb.copy()
        lb[self.i] = self.xi_lb
        return lb
    
    @property
    def ub(self):
        ub = self.parent.ub.copy()
        ub[self.i] = self.xi_ub
        return ub


class RootNode(BaseNode):
    
    children: List[BaseNode]
    
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=0, ub=np.inf,
                 integrality=None, int_tol=1e-6, branching_rule="frac"):
        super().__init__(branching_rule=branching_rule)
        
        # Fix bounds is numeric
        if _check_numeric(lb):
            lb = np.ones_like(c) * lb
        if _check_numeric(ub):
            ub = np.ones_like(c) * ub
        
        # If integrality is None, variables are assumed integer
        if integrality is None:
            integrality = np.ones(len(c), dtype=bool)
        
        # Set remaining values
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.lb = lb
        self.ub = ub
        self.integrality = integrality.astype(bool)
        self.int_tol = int_tol
        self.children = []


def _check_numeric(x: Any):
    return isinstance(x, float) or isinstance(x, int) or isinstance(x, np.integer)
