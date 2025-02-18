# solver/branching_strategies.py

import torch
import math

class BaseBranchingStrategy:
    """Base class for branching strategy. Must implement choose_branch."""
    def __init__(self):
        pass

    def choose_branch(self, solver, lp_solution, patterns):
        """
        solver: reference to the solver instance (can read self.n, demands, etc.)
        lp_solution: list of x_p values from the LP.
        patterns: list of patterns (each a list: pattern[i]=count of item i).
        
        Returns a dict describing the branch decision:
          - For item pair branching: {"type": "pair", "items": (i, j)}
          - For naive variable: {"type": "pattern", "pattern_index": j}
          or None if no branch needed.
        """
        raise NotImplementedError

class NaiveBranchingStrategy(BaseBranchingStrategy):
    """Branches on the first fractional pattern usage variable it finds."""
    def choose_branch(self, solver, lp_solution, patterns):
        for j, val in enumerate(lp_solution):
            if val > 1e-6 and abs(val - round(val)) > 1e-6:
                return {"type": "pattern", "pattern_index": j}
        return None

class RyanFosterStrategy(BaseBranchingStrategy):
    """
    Standard Ryan-Foster branching for cutting stock:
    We pick a pair of items (i, j) that appear fractionally together in patterns.
    Typically, we pick the pair whose combined fraction alpha[i,j] is closest to 0.5
    """
    def choose_branch(self, solver, lp_solution, patterns):
        n = solver.n  # number of item types
        alpha = [[0.0]*n for _ in range(n)]
        # compute alpha[i][j]
        for p_idx, pat in enumerate(patterns):
            x_val = lp_solution[p_idx]
            if x_val < 1e-6:
                continue
            # accumulate
            for i in range(n):
                if pat[i] == 0:
                    continue
                for j in range(i+1, n):
                    if pat[j] == 0:
                        continue
                    alpha[i][j] += x_val
        
        best_pair = None
        best_gap = None
        for i in range(n):
            for j in range(i+1, n):
                if alpha[i][j] > 1e-6 and alpha[i][j] < 1 - 1e-6:
                    gap = abs(alpha[i][j] - 0.5)
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_pair = (i,j)
        
        if best_pair:
            return {"type": "pair", "items": best_pair}
        else:
            # fallback: naive approach if no suitable pair found
            for p_idx, val in enumerate(lp_solution):
                if val > 1e-6 and abs(val - round(val)) > 1e-6:
                    return {"type": "pattern", "pattern_index": p_idx}
            return None

class GNNBranchingStrategy(BaseBranchingStrategy):
    """
    Uses a GNN model to pick either a pattern or item pair. 
    For demonstration, let's pick an item pair from among candidates.
    """
    def __init__(self, gnn_model, state_converter):
        """
        gnn_model: a trained PyTorch model (like BranchingGNN).
        state_converter: a function that turns (solver, lp_solution, patterns) into a GNN input state.
        """
        super().__init__()
        self.model = gnn_model
        self.state_converter = state_converter

    def choose_branch(self, solver, lp_solution, patterns):
        # Build the GNN input from the current node state
        gnn_input, candidate_pairs = self.state_converter(solver, lp_solution, patterns)
        if candidate_pairs is None or len(candidate_pairs)==0:
            # fallback
            for p_idx, val in enumerate(lp_solution):
                if val > 1e-6 and abs(val - round(val))>1e-6:
                    return {"type": "pattern", "pattern_index": p_idx}
            return None
        
        self.model.eval()
        with torch.no_grad():
            scores = self.model(gnn_input)  # e.g. shape [num_candidates]
        # pick argmax
        best_idx = int(torch.argmax(scores))
        chosen_pair = candidate_pairs[best_idx]
        return {"type": "pair", "items": chosen_pair}
