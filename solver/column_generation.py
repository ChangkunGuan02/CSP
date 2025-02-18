# solver/column_generation.py

import gurobipy as gp
from gurobipy import GRB
import math

class ColumnGenerator:
    """
    Handles the pricing subproblem for the Cutting Stock Problem,
    generating patterns (columns) given dual prices. We assume unbounded items for simpler cutting stock, 
    or we can limit item usage if needed.

    Example usage:
      generator = ColumnGenerator(item_lengths, roll_length)
      new_pattern, reduced_cost = generator.generate_pattern(duals)
      if reduced_cost < -1e-8:
          # negative reduced cost => add pattern
    """

    def __init__(self, item_lengths, roll_length, use_ilp=True):
        """
        item_lengths: list of item lengths (integers)
        roll_length: capacity of the stock roll
        use_ilp: if True, use Gurobi ILP for knapsack; else use DP approach.
        """
        self.item_lengths = item_lengths
        self.roll_length = roll_length
        self.n = len(item_lengths)
        self.use_ilp = use_ilp

    def generate_pattern(self, duals):
        """
        Solve the pricing subproblem to find a new pattern with negative reduced cost.
        duals: list of dual prices for each item demand constraint.

        Returns: (pattern, reduced_cost)
          pattern: list of counts of each item in the pattern (size n).
          reduced_cost: 1 - sum(duals[i]*pattern[i]) 
            if < 0 => beneficial column
            if >= 0 => no improvement
        If no improving pattern is found, returns (None, 0).
        """
        if self.use_ilp:
            return self._generate_pattern_ilp(duals)
        else:
            return self._generate_pattern_dp(duals)

    def _generate_pattern_ilp(self, duals):
        """
        Use Gurobi ILP for the knapsack-like subproblem:
          max sum(duals[i]*x_i) - 1
          subject to sum(item_lengths[i]*x_i) <= roll_length
                    x_i >= 0 integer
        """
        try:
            m = gp.Model()
            m.Params.OutputFlag = 0
            x_vars = []
            for i in range(self.n):
                # upper bound can be roll_length // item_lengths[i] if item_lengths[i] > 0
                ub = self.roll_length // self.item_lengths[i] if self.item_lengths[i] > 0 else 0
                var = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, obj=duals[i], name=f"x_{i}")
                x_vars.append(var)
            m.addConstr(gp.quicksum(self.item_lengths[i]*x_vars[i] for i in range(self.n)) <= self.roll_length,
                        name="capacity")
            m.ModelSense = GRB.MAXIMIZE
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                return (None, 0)
            obj_val = m.objVal
            # reduced cost = 1 - obj_val
            # if 1 - obj_val >= 0 => no improving pattern
            if (1 - obj_val) >= -1e-12:  # consider a small tolerance
                return (None, 0)
            # build pattern
            pattern = [int(x_vars[i].X) for i in range(self.n)]
            rc = 1 - obj_val
            return (pattern, rc)
        except gp.GurobiError as e:
            print("Gurobi Error in pricing:", e)
            return (None, 0)

    def _generate_pattern_dp(self, duals):
        """
        A dynamic programming approach for the unbounded knapsack.
        We want to maximize sum(duals[i]*count_i) s.t. sum(item_lengths[i]*count_i) <= roll_length.

        dp[w] = best objective (sum of duals) achievable with capacity w
        We reconstruct the pattern from a backpointer array.
        """
        L = self.roll_length
        dp = [0.0]*(L+1)
        choice = [-1]*(L+1)
        for w in range(L+1):
            dp[w] = 0.0

        for i in range(self.n):
            length_i = self.item_lengths[i]
            if length_i <= 0:
                continue
            for w in range(length_i, L+1):
                val = dp[w - length_i] + duals[i]
                if val > dp[w]:
                    dp[w] = val
                    choice[w] = i
        
        # best_val = max(dp[w]) for w in [0..L]
        best_val = max(dp)
        best_w = dp.index(best_val)
        if (1 - best_val) >= -1e-12:
            return (None, 0)

        # Reconstruct pattern
        pattern = [0]*self.n
        w = best_w
        while w > 0 and choice[w] != -1:
            i = choice[w]
            pattern[i] += 1
            w -= self.item_lengths[i]
        rc = 1 - best_val
        return (pattern, rc)
