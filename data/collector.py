# data/collector.py

import copy
from solver.branch_and_price_solver import CSPBranchAndPriceSolver
from solver.branching_strategies import RyanFosterStrategy, NaiveBranchingStrategy
from ml.graph_utils import build_state_graph_item_pairs
import torch

class ExpertDataCollector:
    """
    Runs the solver with an expert branching strategy, hooking into the branch function 
    to record (gnn_input, candidate_pairs, label).
    """

    def __init__(self, expert_strategy="ryan-foster"):
        if expert_strategy=="ryan-foster":
            self.expert_strat = RyanFosterStrategy()
        elif expert_strategy=="naive":
            self.expert_strat = NaiveBranchingStrategy()
        else:
            raise ValueError("Unknown expert strategy.")
        self.collected_data = []

    def collect_data_from_instance(self, instance):
        """
        Returns a list of samples from a single CSP instance. 
        Each sample is a dict with gnn_input, candidate_pairs, label.
        """
        self.collected_data = []
        # We'll define a special solver that intercepts the 'choose_branch' call
        solver = HookedCSPBranchAndPriceSolver(
            instance["roll_length"],
            instance["item_lengths"],
            instance["demands"],
            self.expert_strat,
            self
        )
        solver.solve()
        return self.collected_data

class HookedCSPBranchAndPriceSolver(CSPBranchAndPriceSolver):
    """
    Extends CSPBranchAndPriceSolver to intercept the branching decision 
    and record data for supervised training.
    """
    def __init__(self, roll_length, item_lengths, demands, expert_strat, collector):
        super().__init__(roll_length, item_lengths, demands, expert_strat, logger=None)
        self.collector = collector
    
    def _solve_node_lp(self, node):
        # same column generation as base
        return super()._solve_node_lp(node)

    def _branch_and_price_node(self, node):
        if self.node_count >= 1e9:
            return
        self.node_count += 1
        final_obj, lp_solution = self._solve_node_lp(node)
        # check integrality, prune, etc. same as base
        # ...
        fractional_indices = [j for j, val in enumerate(lp_solution)
                              if val>1e-6 and abs(val-round(val))>1e-6]
        if len(fractional_indices)==0:
            if final_obj < self.best_int_obj:
                self.best_int_obj = final_obj
                self.best_solution = (node.patterns, lp_solution)
            return
        # Intercept the branching
        decision = self.branching_strategy.choose_branch(self, lp_solution, node.patterns)
        # here we record the state => label
        if decision:
            if decision["type"]=="pair":
                # we build gnn input
                gnn_input, candidate_pairs = build_state_graph_item_pairs(self, lp_solution, node.patterns)
                # find which pair was chosen
                chosen = decision["items"]
                # find index
                c_idx = None
                for idx, pair in enumerate(candidate_pairs):
                    if pair==chosen or (pair==(chosen[1],chosen[0])):
                        c_idx = idx
                        break
                if c_idx is not None:
                    sample = {
                        "gnn_input": gnn_input,
                        "candidate_pairs": candidate_pairs,
                        "label": c_idx
                    }
                    self.collector.collected_data.append(sample)
            else:
                # branching on pattern => we won't store it 
                # or we can store it as a different label system
                pass
        
        if decision is None:
            return
        # proceed to create child nodes
        from solver.node import copy_node, SearchNode
        new_node = SearchNode(node.patterns, node.constraints, node.depth, final_obj)
        if decision["type"]=="pair":
            (i,j) = decision["items"]
            c1 = copy_node(new_node)
            c1.depth += 1
            c1.constraints.append({"type":"pair","items":(i,j),"together":True})
            c2 = copy_node(new_node)
            c2.depth += 1
            c2.constraints.append({"type":"pair","items":(i,j),"together":False})
            self._branch_and_price_node(c1)
            self._branch_and_price_node(c2)
        elif decision["type"]=="pattern":
            p_idx = decision["pattern_index"]
            c1 = copy_node(new_node)
            c1.depth += 1
            c1.constraints.append({"type":"pattern","pattern_index":p_idx,"must_use":False})
            c2 = copy_node(new_node)
            c2.depth += 1
            c2.constraints.append({"type":"pattern","pattern_index":p_idx,"must_use":True})
            self._branch_and_price_node(c1)
            self._branch_and_price_node(c2)


def collect_dataset(instances, expert_strategy="ryan-foster"):
    """
    Runs the expert solver on each instance, collecting training data for supervised learning.
    Returns a list of samples. 
    """
    collector = ExpertDataCollector(expert_strategy)
    dataset = []
    for inst in instances:
        samples = collector.collect_data_from_instance(inst)
        dataset.extend(samples)
    return dataset
