# ml/dataset.py

import torch
from torch.utils.data import Dataset

class BranchingDataset(Dataset):
    """
    Expects data as a list of dicts:
      each dict has:
        'gnn_input': dict with item_feat, pattern_feat, edge_index_item_to_pattern, etc.
        'candidate_pairs': list of (i,j)
        'label': int (index in candidate_pairs that the expert chose)
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        return sample

def collate_fn(batch):
    """
    Collate function if we want to handle variable-size graphs in a single batch.
    For simplicity, we'll just return them as a list. The training loop can handle them 1 by 1.
    """
    return batch
