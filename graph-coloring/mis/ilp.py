from typing import List, Tuple

import pyomo.environ as pyo

from mis.graph import Graph


# Do not repeat colors on edges and color is used
def edge_cstr(model, i, j):
    return model.x[i] + model.x[j] <= 1


def ilp_mis_from_graph(graph: Graph) -> pyo.ConcreteModel:
    """Instantiates pyomo Integer Linear Programming model for the Maximum Independent Set Problem

    Parameters
    ----------
    graph : Graph
        Graph with partial solution

    Returns
    -------
    pyo.ConcreteModel
        `Concretemodel` of pyomo
    """
    nodes = [i for i in graph.N.keys()]
    edges = [(n.index, m.index) for n in graph.N.values() for m in n.neighbors]
    model = build_mis_ilp(nodes, edges)
    warmstart_from_greedy(model, graph)
    return model


def build_mis_ilp(
    nodes: List[int],
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
    model.N = pyo.Set(initialize=nodes)  # Nodes
    model.E = pyo.Set(initialize=edges)  # Edges

    # Create variables
    model.x = pyo.Var(model.N, within=pyo.Binary)

    # Create constraints
    model.edge_cstr = pyo.Constraint(model.E, rule=edge_cstr)

    # Create objective
    model.obj = pyo.Objective(expr=sum(model.x[:]), sense=pyo.maximize)

    return model


def warmstart_from_greedy(model, graph: Graph):
    for n in graph.N.values():
        if n.selected:
            model.x[n.index].value = 1.0
        else:
            model.x[n.index].value = 0.0
