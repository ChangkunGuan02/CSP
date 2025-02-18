# evaluation/evaluate_solver.py

import time
import csv
import os
from solver.branch_and_price_solver import CSPBranchAndPriceSolver
from solver.branching_strategies import NaiveBranchingStrategy, RyanFosterStrategy
# from ml.graph_utils import build_state_graph_item_pairs -> might be needed if GNN strategy
# from ml.model import BranchingGNN, ItemPairBranchModel -> load your model
# from ml.graph_utils import build_state_graph_item_pairs
# from solver.branching_strategies import GNNBranchingStrategy

def evaluate_strategies(instances, strategies, output_csv="strategy_comparison.csv"):
    """
    instances: list of CSP instances
    strategies: dict of { strategy_name: branching_strategy_object }
    output_csv: path to store results
    We measure time, node_count, and final objective.
    """
    results = []
    for idx, inst in enumerate(instances):
        for strat_name, strat_obj in strategies.items():
            start_t = time.time()
            solver = CSPBranchAndPriceSolver(inst["roll_length"], inst["item_lengths"],
                                             inst["demands"], strat_obj, logger=None)
            best_obj, _ = solver.solve()
            end_t = time.time()
            run_time = end_t - start_t
            results.append({
                "instance_id": idx,
                "strategy": strat_name,
                "time": run_time,
                "nodes": solver.node_count,
                "best_obj": best_obj
            })
    # write to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id","strategy","time","nodes","best_obj"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return results
