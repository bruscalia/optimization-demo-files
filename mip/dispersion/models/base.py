import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import pyomo.environ as pyo


def p_selection(model):
    return sum(model.x[:]) == model.p


class DispersionModel(pyo.ConcreteModel):

    coordinates: np.ndarray

    def __init__(self, coordinates: np.ndarray, p: int):
        """Facility location model

        To define a custom distance evaluation function, please overwrite the method `distance_calc`

        Parameters
        ----------
        coordinates : np.ndarray
            Coordinates of possible locations (N, M) in which N is the number of candidates and M
            is the number of dimensions

        p : int
            Number of facilities selectes
        """
        super().__init__()

        # Basic attributes
        self.coordinates = coordinates
        distances = self.distance_calc()
        N = distances.shape[0]

        # Sets
        self.V = pyo.Set(initialize=range(N))
        self.A = pyo.Set(initialize=[(i, j) for i in self.V for j in self.V if i != j])

        # Parameters
        self.d = pyo.Param(self.A, initialize={(i, j): distances[i, j] for (i, j) in self.A})
        self.p = pyo.Param(initialize=p)

        # Decision variables
        self.x = pyo.Var(self.V, within=pyo.Binary)

        # Constraints
        self.p_selection = pyo.Constraint(rule=p_selection)

    def distance_calc(self):
        return squareform(pdist(self.coordinates))

    def plot(self, figsize=[6, 5], dpi=100, base_color="navy", selected_color="firebrick"):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        facilities = np.array([i for i in self.x if np.isclose(self.x[i].value, 1)])
        ax.scatter(
            self.coordinates[:, 0],
            self.coordinates[:, 1],
            color=base_color
        )
        ax.scatter(
            self.coordinates[facilities, 0],
            self.coordinates[facilities, 1],
            color=selected_color,
            label="Facilities"
        )
        ax.legend()
        fig.tight_layout()
        return fig