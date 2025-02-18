# ml/train_rl.py

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import math
from solver.branch_and_price_solver import CSPBranchAndPriceSolver

def fine_tune_with_reinforcement(model, instances, solver_params,
                                 episodes=50, gamma=0.99, lr=1e-4,
                                 log_dir="runs/rl", save_path="rl_model.pt"):
    """
    model: an ItemPairBranchModel (or similar),
    instances: list of CSP instances, each a dict with {roll_length, item_lengths, demands}
    solver_params: dict of parameters for the CSPBranchAndPriceSolver except branching_strategy
                   e.g. {'logger': None, 'use_ilp_pricing': False}
    We use a simple REINFORCE approach: 
      - at each node, the solver calls our GNN strategy (with exploration).
      - after the instance is solved, we get a final reward = -(node_count).
    """
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0

    for ep in range(episodes):
        # pick an instance
        inst = random.choice(instances)
        # create a custom branching strategy that calls the model in a training mode
        # we'll define a local RL strategy class inline
        rl_strategy = RLBranchingStrategy(model)
        solver = CSPBranchAndPriceSolver(
            inst["roll_length"], inst["item_lengths"], inst["demands"],
            branching_strategy=rl_strategy,
            **solver_params
        )
        # We do the solve, which will record a trajectory internally
        solver.solve()
        # node_count = solver.node_count, we define reward = -node_count
        R = -solver.node_count
        # now update model with REINFORCE
        returns = 0
        for (log_prob) in reversed(rl_strategy.log_probs):
            returns = R + gamma*returns
            # no baseline for simplicity
            loss = -log_prob * returns
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        writer.add_scalar("EpisodeReward", R, ep)
        print(f"Episode {ep+1}/{episodes}, Reward={R}, nodes={solver.node_count}")

    torch.save(model.state_dict(), save_path)
    writer.close()

class RLBranchingStrategy:
    """
    A branching strategy that uses a GNN model in a RL setting.
    We store (log_prob) for each decision. The final reward is assigned after the instance is done.
    """
    def __init__(self, model):
        self.model = model
        self.log_probs = []

    def choose_branch(self, solver, lp_solution, patterns):
        # convert state
        from ml.graph_utils import build_state_graph_item_pairs
        gnn_input, candidate_pairs = build_state_graph_item_pairs(solver, lp_solution, patterns)
        if not candidate_pairs:
            # fallback to naive
            for p_idx, val in enumerate(lp_solution):
                if val>1e-6 and abs(val-round(val))>1e-6:
                    return {"type":"pattern", "pattern_index": p_idx}
            return None

        self.model.train()
        scores = self.model(gnn_input, candidate_pairs)  # shape [num_candidates]
        probs = F.softmax(scores, dim=0)
        # sample from distribution
        c = torch.multinomial(probs, 1).item()
        log_p = torch.log(probs[c])
        self.log_probs.append(log_p)
        chosen_pair = candidate_pairs[c]
        return {"type":"pair", "items":chosen_pair}
