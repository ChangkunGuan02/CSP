# solver/branch_and_price_solver.py

import gurobipy as gp
from gurobipy import GRB
import math
import copy

from .node import SearchNode, copy_node
from .logger import SolverLogger
from .column_generation import ColumnGenerator

class CSPBranchAndPriceSolver:
    """
    Branch-and-price solver for the Cutting Stock Problem using Gurobi.
    Integrates column generation with branching strategies. 
    """

    def __init__(self, roll_length, item_lengths, demands, branching_strategy,
                 logger=None, use_ilp_pricing=False):
        """
        roll_length: capacity of each stock roll
        item_lengths: list of item lengths
        demands: list of demands for each item type
        branching_strategy: an instance of BaseBranchingStrategy (naive, ryan-foster, or GNN)
        logger: optional SolverLogger for structured logging
        use_ilp_pricing: if True, column generation uses ILP for knapSack; else DP.
        """
        self.L = roll_length
        self.item_lengths = item_lengths
        self.demands = demands
        self.n = len(item_lengths)
        self.branching_strategy = branching_strategy
        self.logger = logger
        self.use_ilp_pricing = use_ilp_pricing

        self.best_int_obj = float('inf')
        self.best_solution = None  # (patterns, x_vals) or some representation
        self.node_count = 0
        self.pattern_counter = 0  # to label patterns if needed

    def solve(self, max_nodes=1e9):
        """
        Public entry point. Creates root node, then does a search. Returns best integer solution objective.
        """
        if self.logger:
            self.logger.open()
            self.logger.log_event("SolverStart", 0, 0, "Starting Branch-and-Price solver")

        # build initial patterns: trivial patterns (one item per pattern)
        initial_patterns = []
        for i in range(self.n):
            pat = [0]*self.n
            pat[i] = 1
            initial_patterns.append(pat)

        root_node = SearchNode(
            patterns = initial_patterns,
            constraints = [],
            depth = 0,
            lp_obj = None
        )

        self.node_stack = []
        self._branch_and_price_node(root_node)

        if self.logger:
            self.logger.log_event("SolverEnd", 0, 0, f"BestObj={self.best_int_obj}")
            self.logger.close()

        return self.best_int_obj, self.best_solution

    def _branch_and_price_node(self, node):
        if self.node_count >= 1e9:
            # guard to prevent infinite loops
            return
        self.node_count += 1
        
        # Column generation for this node
        final_obj, lp_solution = self._solve_node_lp(node)
        new_node = SearchNode(
            patterns=node.patterns,
            constraints=node.constraints,
            depth=node.depth,
            lp_obj=final_obj
        )

        if self.logger:
            self.logger.log_event("NodeSolved", node.depth, final_obj,
                                  f"NodeID={self.node_count}, patterns={len(node.patterns)}")

        # bound check
        if final_obj >= self.best_int_obj - 1e-9:
            # prune
            if self.logger:
                self.logger.log_event("Prune", node.depth, final_obj, "LP bound >= best int obj")
            return

        # check integrality
        fractional_indices = [j for j, val in enumerate(lp_solution) 
                              if val>1e-6 and abs(val-round(val))>1e-6]
        if len(fractional_indices) == 0:
            # integral => update best solution
            if final_obj < self.best_int_obj - 1e-9:
                self.best_int_obj = final_obj
                self.best_solution = (node.patterns, lp_solution)
            if self.logger:
                self.logger.log_event("IntegralFound", node.depth, final_obj, f"NodeID={self.node_count}")
            return

        # Need to branch
        decision = self.branching_strategy.choose_branch(self, lp_solution, node.patterns)
        if decision is None:
            # no branching => weird case, but let's just stop
            return
        if self.logger:
            self.logger.log_event("Branch", node.depth, final_obj, str(decision))

        # create child nodes
        if decision["type"] == "pair":
            (i, j) = decision["items"]
            # Child1: items i,j forced together
            child1 = copy_node(new_node)
            child1.depth += 1
            child1.constraints.append({"type":"pair", "items":(i,j), "together":True})

            # Child2: items i,j forced apart
            child2 = copy_node(new_node)
            child2.depth += 1
            child2.constraints.append({"type":"pair", "items":(i,j), "together":False})

            self._branch_and_price_node(child1)
            self._branch_and_price_node(child2)

        elif decision["type"] == "pattern":
            p_idx = decision["pattern_index"]
            # Child1: pattern p_idx = 0
            child1 = copy_node(new_node)
            child1.depth += 1
            child1.constraints.append({"type":"pattern", "pattern_index":p_idx, "must_use":False})
            # Child2: pattern p_idx >= 1
            child2 = copy_node(new_node)
            child2.depth += 1
            child2.constraints.append({"type":"pattern", "pattern_index":p_idx, "must_use":True})
            self._branch_and_price_node(child1)
            self._branch_and_price_node(child2)

    def _solve_node_lp(self, node):
        """
        Solve the master LP at this node with column generation,
        returning the final LP objective and the solution vector x.
        """
        patterns = node.patterns
        constraints = node.constraints

        # column generation
        while True:
            lp_obj, x_vals, duals = self._solve_lp_relaxation(patterns, constraints)
            # pricing
            generator = ColumnGenerator(self.item_lengths, self.L, use_ilp=self.use_ilp_pricing)
            new_pattern, rc = generator.generate_pattern(duals)
            if new_pattern is None:
                break
            if rc >= -1e-10:
                break
            patterns.append(new_pattern)

        # final solve to get fresh x_vals
        lp_obj, x_vals, duals = self._solve_lp_relaxation(patterns, constraints)
        return lp_obj, x_vals

    def _solve_lp_relaxation(self, patterns, constraints):
        """
        Build and solve the LP with Gurobi. 
        patterns: list of columns
        constraints: list of branch constraints
        Returns: (obj, x_vals, duals).
        duals: list of dual prices for each item demand.
        """
        m = gp.Model()
        m.Params.OutputFlag = 0  # silent
        x_vars = []
        for j, pat in enumerate(patterns):
            var = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=1.0, name=f"x_{j}")
            x_vars.append(var)

        # demand constraints
        demand_constrs = []
        for i, dem in enumerate(self.demands):
            c = m.addConstr(gp.quicksum(pat[i]*x_vars[j] for j, pat in enumerate(patterns)) >= dem, 
                            name=f"demand_{i}")
            demand_constrs.append(c)

        # branching constraints
        for bc in constraints:
            if bc["type"] == "pair":
                (i, j) = bc["items"]
                if bc["together"]:
                    # forbid patterns that have i but not j or j but not i
                    for idx, pat in enumerate(patterns):
                        if (pat[i] > 0 and pat[j] == 0) or (pat[j] > 0 and pat[i] == 0):
                            m.addConstr(x_vars[idx] == 0, name=f"together_forbid_{idx}_{i}_{j}")
                else:
                    # apart => forbid patterns that have both i and j
                    for idx, pat in enumerate(patterns):
                        if pat[i] > 0 and pat[j] > 0:
                            m.addConstr(x_vars[idx] == 0, name=f"apart_forbid_{idx}_{i}_{j}")
            elif bc["type"] == "pattern":
                p_idx = bc["pattern_index"]
                must_use = bc["must_use"]
                if p_idx < len(x_vars):
                    if must_use:
                        m.addConstr(x_vars[p_idx] >= 1, name=f"must_use_{p_idx}")
                    else:
                        m.addConstr(x_vars[p_idx] == 0, name=f"forbid_{p_idx}")

        m.ModelSense = GRB.MINIMIZE
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return (float('inf'), [], [])

        obj_val = m.ObjVal
        x_sol = [v.X for v in x_vars]
        # get duals for demand constraints
        duals = [c.Pi for c in demand_constrs]
        return (obj_val, x_sol, duals)
