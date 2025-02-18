# scripts/evaluate_supervised.py

import torch
from torch.utils.data import DataLoader
from ml.model import BranchingGNN, ItemPairBranchModel
from ml.dataset import BranchingDataset, collate_fn
import torch.nn.functional as F

def main():
    # load test dataset
    test_data = torch.load("test_data.pt")  # assume you have it
    test_ds = BranchingDataset(test_data)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # load model
    gnn_base = BranchingGNN(item_feat_dim=2, pattern_feat_dim=3, hidden_dim=32, message_passing_rounds=2)
    model = ItemPairBranchModel(gnn_base)
    model.load_state_dict(torch.load("supervised_model.pt"))
    model.eval()

    # compute accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            for sample in batch:
                gnn_input = sample["gnn_input"]
                cand_pairs = sample["candidate_pairs"]
                label = sample["label"]
                scores = model(gnn_input, cand_pairs)
                pred = int(torch.argmax(scores))
                if pred == label:
                    correct += 1
                total += 1
    acc = float(correct)/float(total) if total>0 else 0.0
    print(f"Test Accuracy = {acc*100:.2f}%")

if __name__=="__main__":
    main()
