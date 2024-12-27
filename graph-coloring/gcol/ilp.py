from typing import List, Tuple

import pyomo.environ as pyo

from gcol.datamodels.graph import Graph


# Fill every node with some color
def fill_cstr(model, i):
    return sum(model.x[i, :]) == 1


# Do not repeat colors on edges and color is used
def edge_cstr(model, i, j, c):
    return model.x[i, c] + model.x[j, c] <= model.y[c]


# Break symmetry by setting a preference order
def break_symmetry(model, c):
    if model.C.first() == c:
        return 0 <= model.y[c]
    else:
        c_prev = model.C.prev(c)
        return model.y[c] <= model.y[c_prev]


# Total number of colors used
def obj(model):
    return sum(model.y[:])


def ilp_from_graph(graph: Graph) -> pyo.ConcreteModel:
    """
    Instantiates pyomo Integer Linear Programming
    model for the Graph Coloring Problem

    Parameters
    ----------
    graph : Graph
        Graph representation of solved instance

    Returns
    -------
    pyo.ConcreteModel
        `Concretemodel` of pyomo
    """
    nodes = [n.index for n in graph.nodes.values()]
    colors = [c.index for c in graph.colors]
    edges = [
        (n.index, m.index)
        for n in graph.nodes.values()
        for m in n.neighbors
    ]
    model = build_ilp(nodes, colors, edges)
    warmstart_from_graph(model, graph)
    return model


def build_ilp(
    nodes: List[int],
    colors: List[int],
    edges: List[Tuple[int, int]]
) -> pyo.ConcreteModel:
    """Instantiates pyomo Integer Linear Programming model for the Graph Coloring Problem

    Parameters
    ----------
    nodes : List[int]
        Node indexes

    colors : List[int]
        List of available colors

    edges : List[Tuple[int, int]]
        Connected edges

    Returns
    -------
    pyo.ConcreteModel
        `Concretemodel` of pyomo
    """

    # Create instance
    model = pyo.ConcreteModel()

    # Create sets
    model.C = pyo.Set(initialize=colors)  # Colors
    model.N = pyo.Set(initialize=nodes)  # Nodes
    model.E = pyo.Set(initialize=edges)  # Edges

    # Create variables
    model.x = pyo.Var(model.N, model.C, within=pyo.Binary)
    model.y = pyo.Var(model.C, within=pyo.Binary)

    # Create constraints
    model.fill_cstr = pyo.Constraint(model.N, rule=fill_cstr)
    model.edge_cstr = pyo.Constraint(model.E, model.C, rule=edge_cstr)
    model.break_symmetry = pyo.Constraint(model.C, rule=break_symmetry)

    # Create objective
    model.obj = pyo.Objective(rule=obj)

    return model


def warmstart_from_graph(model, graph: Graph):
    for n in graph.nodes.values():
        for c in graph.colors:
            if n.color is c:
                model.x[n.index, c.index].value = 1.0
            else:
                model.x[n.index, c.index].value = 0.0

    for c in graph.colors:
        model.y[c.index].value = 1.0
