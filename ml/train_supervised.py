# ml/train_supervised.py

import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from .dataset import BranchingDataset, collate_fn

def train_gnn_supervised(model, training_data, val_data=None, 
                         lr=1e-3, epochs=20, batch_size=1, 
                         log_dir="runs/supervised", save_path="supervised_model.pt"):
    """
    training_data, val_data: list of samples (gnn_input, candidate_pairs, label)
    Each sample is a dict:
      {
        'gnn_input': {...},
        'candidate_pairs': [...],
        'label': int
      }
    model: an ItemPairBranchModel or similar
    
    We'll do basic cross-entropy over candidate pairs.
    """
    writer = SummaryWriter(log_dir=log_dir)
    train_dataset = BranchingDataset(training_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    if val_data is not None:
        val_dataset = BranchingDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            # batch is a list of samples
            # process each sample individually or find a way to batch them (if possible).
            # here, we do them one by one.
            for sample in batch:
                gnn_input = sample['gnn_input']
                cand_pairs = sample['candidate_pairs']
                label = sample['label']

                optimizer.zero_grad()
                scores = model(gnn_input, cand_pairs)  # shape [num_candidates]
                # cross-entropy
                scores_2d = scores.unsqueeze(0)  # [1, num_candidates]
                target = torch.tensor([label], dtype=torch.long)
                loss = F.cross_entropy(scores_2d, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_samples += 1
                global_step += 1
        
        avg_loss = total_loss / max(1, total_samples)
        writer.add_scalar("Train/Loss", avg_loss, epoch)

        # Evaluate on val set, if any
        if val_loader is not None:
            val_acc = evaluate_accuracy(model, val_loader)
            writer.add_scalar("Val/Accuracy", val_acc, epoch)
            print(f"Epoch {epoch+1}/{epochs}, TrainLoss={avg_loss:.4f}, ValAcc={val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, TrainLoss={avg_loss:.4f}")

    # save final model
    torch.save(model.state_dict(), save_path)
    writer.close()


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            for sample in batch:
                gnn_input = sample['gnn_input']
                cand_pairs = sample['candidate_pairs']
                label = sample['label']
                scores = model(gnn_input, cand_pairs)
                pred = int(torch.argmax(scores))
                if pred == label:
                    correct += 1
                total += 1
    acc = float(correct) / float(total) if total>0 else 0.0
    return acc
