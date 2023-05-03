from typing import List, Union

import numpy as np
from scipy.optimize import OptimizeResult

from bnb.nodes import Node, RootNode, BaseNode


class BranchAndBound:
    
    def __init__(self, branching_rule="min", node_rule="dfs"):
        """Branch and Bound basic algorithm

        Parameters
        ----------
        branching_rule : str, optional
            How to define in which variable to branch on, by default "min"
            Options are:
                - "min": branching on the smallest violation to floor
                - "max": branching on the largest violation to ceil
                - "frac": branching on the most fractional value
        
        node_rule : str, optional
            How to explore the search tree, by default "dfs"
            Options are:
            - "dfs": Depth first search
            - "bfs": Breadth first search
        """
        self.branching_rule = branching_rule
        self.node_rule = node_rule
    
    def __call__(
        self,
        c: np.ndarray,
        A_ub: np.ndarray = None,
        b_ub: np.ndarray = None,
        A_eq: np.ndarray = None,
        b_eq: np.ndarray = None,
        lb: Union[int, float, np.ndarray] = None,
        ub: Union[int, float, np.ndarray] = None,
        integrality: np.ndarray = None,
        mip_gap: float = 1e-4,
        int_tol: float = 1e-4,
        max_iter: int = 1000,
        verbose: bool = False,
    ) -> OptimizeResult:
        """Solve a MILP

        Parameters
        ----------
        c : np.ndarray
            Cost associated with each decision variable
        
        A_ub : np.ndarray, optional
            Matrix of inequality constraints, by default None
        
        b_ub : np.ndarray, optional
            Array of inequality RHS values, by default None
        
        A_eq : np.ndarray, optional
            Matrix of equality constraints, by default None
        
        b_eq : np.ndarray, optional
            Array of equality RHS values, by default None
        
        lb : Union[int, float, np.ndarray], optional
            Array or scalar of lower bounds, by default None
        
        ub : Union[int, float, np.ndarray], optional
            Array or scalar of upper bounds, by default None
        
        integrality : np.ndarray, optional
            Integrality boolean array, by default None
        
        mip_gap : float, optional
            MIP tolerance gap, by default 1e-4
        
        int_tol : float, optional
            Integrality tolerance, by default 1e-4
        
        max_iter : int, optional
            Maximum number of iterations, by default 1000
        
        verbose : bool, optional
            Either or not to print display messages, by default False

        Returns
        -------
        OptimizeResult
            Scipy optimize results
        """
        
        # Define type of queue
        queue: List[Node]
        
        # Start at root node
        root_node = RootNode(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub,
            integrality=integrality, branching_rule=self.branching_rule, 
            int_tol=int_tol,
        )
        root_node.solve()
        
        # Initialize values
        best_bound = root_node.sol.fun
        incumbent = np.inf
        is_optimal = root_node.integer
        message = "No solution found"
        sol = OptimizeResult(fun=incumbent, x=None, message=message)
        queue = []
        
        # Check if root node has integer solution
        if not is_optimal and root_node.feasible:
            if verbose:
                print("First solution is feasible but not integer")
            message = "Feasible relaxation but no incumbent"
            root_node.branch()
            queue.extend(root_node.children)
        elif is_optimal:
            sol = root_node.sol
            message = "Integer optimal solution on root node"
        else:
            return OptimizeResult(success=False, message="Infeasible")
        
        # Count nodes explored
        k = 0
        
        # Iterate until all good nodes are explored
        while len(queue) > 0 and k < max_iter and best_bound < incumbent:
            
            # Check relaxation of parents -> Ineficient
            best_bound = min(n.parent.sol.fun for n in queue)
            
            # Break if new best bound reduces gap enough
            gap = self._calc_gap(best_bound, incumbent)
            if gap < mip_gap:
                message = f"MIP gap tolance reached {gap:.4f}"
                break

            # Choose next node
            node = self._pop_next(queue)
            
            # Check if node is still worth solving
            if node.parent.sol.fun > incumbent:
                if verbose:
                    print(f"Fathom node {k} due to poor parent node relaxation")
                continue
            else:
                k = k + 1
                node.solve()
            
            # Check if feasible
            if not node.feasible:
                if verbose:
                    print(f"Infeasible node {k}")
                continue
            
            # Feasible solution
            elif node.integer:
                if node.sol.fun < incumbent:
                    sol = node.sol
                    incumbent = node.sol.fun
                    if verbose:
                        print(f"New best sol {k}: {node.sol.fun}")
                else:
                    if verbose:
                        print(f"Integer but not the best {k}: {node.sol.fun}")
                    continue
            
            # Not integer solution
            elif node.sol.fun < incumbent:
                if verbose:
                    print(f"Feasible below incumbent {k}: {node.sol.fun}")
                node.branch()
                queue.extend(node.children)
                continue
            
            else:
                if verbose:
                    print(f"Feasible above incumbent -> Fathom {k}: {node.sol.fun}")
                continue
        
        # queue is empty
        if len(queue) == 0 and sol.status == 0:
            gap = 0
            best_bound = incumbent
            message = "Optimal integer solution found"
        
        sol.message = message
        sol.nodes = k
        sol.best_bound = best_bound
        sol.mip_gap = gap
        return sol
    
    def _pop_next(self, queue: List[BaseNode]):
        if self.node_rule == "dfs":
            node = queue.pop(-1)
        elif self.node_rule == "bfs":
            node = queue.pop(0)
        else:
            raise ValueError("node_rule must be either 'dfs' or 'bfs'")
        return node
    
    def _calc_gap(self, best_bound, incumbent):
        if incumbent != np.inf:
            return abs(best_bound - incumbent) / max(abs(best_bound), abs(incumbent), 1)
        else:
            return np.inf
