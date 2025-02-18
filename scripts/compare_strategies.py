# scripts/compare_strategies.py

import torch
from solver.branching_strategies import NaiveBranchingStrategy, RyanFosterStrategy, GNNBranchingStrategy
from evaluation.evaluate_solver import evaluate_strategies
from ml.graph_utils import build_state_graph_item_pairs
from ml.model import BranchingGNN, ItemPairBranchModel
import os

def main():
    # load test instances
    import pickle
    with open("test_instances.pkl","rb") as f:
        instances = pickle.load(f)

    # define naive and ryan-foster
    naive = NaiveBranchingStrategy()
    ryan = RyanFosterStrategy()

    # define GNN-based
    gnn_base = BranchingGNN(item_feat_dim=2, pattern_feat_dim=3, hidden_dim=32)
    model = ItemPairBranchModel(gnn_base)
    model.load_state_dict(torch.load("supervised_model.pt"))
    def state_converter(solver, lp_solution, patterns):
        return build_state_graph_item_pairs(solver, lp_solution, patterns)
    gnn_strat = GNNBranchingStrategy(model, state_converter)

    strategies = {
        "naive": naive,
        "ryan-foster": ryan,
        "gnn": gnn_strat
    }

    # evaluate
    results = evaluate_strategies(instances, strategies, output_csv="strategy_comparison.csv")
    print("Evaluation done. See strategy_comparison.csv")

if __name__=="__main__":
    main()
