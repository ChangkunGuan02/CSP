# ml/training.py

import torch
import torch.nn.functional as F

def train_gnn_supervised(model, optimizer, training_data, epochs=20):
    """
    training_data: list of (state, expert_decision_index) tuples
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for state, expert_index in training_data:
            optimizer.zero_grad()
            scores = model(state)  # [num_vars] 
            # Cross-entropy with the expert's chosen var index
            target = torch.tensor([expert_index], dtype=torch.long)
            scores_2d = scores.view(1, -1) 
            loss = F.cross_entropy(scores_2d, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}, average loss = {avg_loss:.4f}")


def fine_tune_with_reinforcement(model, optimizer, instances, solver_class, gamma=0.99):
    """
    'instances' is a list of problem instances, each with roll_length, item_lengths, demands, etc.
    'solver_class' is something like CuttingStockBPSolver
    """
    model.train()
    for inst in instances:
        # Create solver with ML-based branching
        solver = solver_class(inst.roll_length, inst.item_lengths, inst.demands,
                              branching_strategy="ml", ml_model=model)
        solver.solve()  # user-defined method that runs branch_and_price_node at root

        # Suppose the solver logs states/actions in solver.collected_trajectory
        trajectory = solver.collected_trajectory  # e.g. [(state, action, reward), ...]

        # Calculate discounted returns
        returns = []
        G = 0
        for (_, _, reward) in reversed(trajectory):
            G = reward + gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        optimizer.zero_grad()
        loss = 0.0
        for (state, action_idx, _), R in zip(trajectory, returns):
            scores = model(state)
            log_probs = F.log_softmax(scores, dim=0)
            loss -= log_probs[action_idx] * R
        loss.backward()
        optimizer.step()
