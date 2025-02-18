# scripts/train_gnn.py

import torch
from ml.model import BranchingGNN, ItemPairBranchModel
from ml.train_supervised import train_gnn_supervised

def main():
    # load data
    dataset = torch.load("supervised_data.pt")  # list of dicts

    # define GNN model
    gnn_base = BranchingGNN(item_feat_dim=2, pattern_feat_dim=3, hidden_dim=32, message_passing_rounds=2)
    model = ItemPairBranchModel(gnn_base)

    # split train/val
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    train_gnn_supervised(
        model, train_data, val_data=val_data,
        lr=1e-3, epochs=10, batch_size=1,
        log_dir="runs/supervised",
        save_path="supervised_model.pt"
    )

if __name__=="__main__":
    main()
