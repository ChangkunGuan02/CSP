# scripts/generate_data.py

import torch
import os
from data.generator import generate_multiple_instances
from data.collector import collect_dataset


def main():
    # 1) Generate random instances
    instances = generate_multiple_instances(
        count=10,
        num_items=8,
        roll_length=50,
        min_item_length=5,
        max_item_length=50,
        min_demand=2,
        max_demand=6
    )

    # 2) Collect supervised data from an expert
    dataset = collect_dataset(instances, expert_strategy="ryan-foster")
    print(f"Collected {len(dataset)} training samples.")

    # 3) Save to .pt
    torch.save(dataset, "supervised_data.pt")
    print("Saved dataset to supervised_data.pt")

if __name__=="__main__":
    main()
