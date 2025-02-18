# ml/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchingGNN(nn.Module):
    """
    A bipartite GNN that processes item and pattern nodes, 
    then produces a score for a list of candidate pairs or variables. 
    For item pairs, we do an extra step to compute pair embeddings from item embeddings.
    """
    def __init__(self, item_feat_dim=2, pattern_feat_dim=3, hidden_dim=32, message_passing_rounds=2):
        super().__init__()
        self.item_feat_dim = item_feat_dim
        self.pattern_feat_dim = pattern_feat_dim
        self.hidden_dim = hidden_dim
        self.message_passing_rounds = message_passing_rounds

        # initial embeddings
        self.item_embed = nn.Linear(item_feat_dim, hidden_dim)
        self.pattern_embed = nn.Linear(pattern_feat_dim, hidden_dim)
        # message transforms
        self.item_to_pattern = nn.Linear(hidden_dim, hidden_dim)
        self.pattern_to_item = nn.Linear(hidden_dim, hidden_dim)

        # final pair scorer: given (hi, hj) => produce a scalar
        self.pair_scorer = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        state: dict with:
           'item_feat' -> [n, item_feat_dim]
           'pattern_feat' -> [m, pattern_feat_dim]
           'edge_index_item_to_pattern' -> list of (i, p)
           'candidate_pairs' -> optional if we incorporate it here
        We'll assume we only return item embeddings, and an external function 
        computes pair embeddings for candidate pairs. 
        For brevity, we just handle item embeddings. 
        """
        item_feat = state["item_feat"]
        pattern_feat = state["pattern_feat"]
        edges = state["edge_index_item_to_pattern"]

        n = item_feat.size(0)
        m = pattern_feat.size(0)

        h_item = F.relu(self.item_embed(item_feat))    # [n, hidden_dim]
        h_pat = F.relu(self.pattern_embed(pattern_feat))  # [m, hidden_dim]

        # message passing
        for _ in range(self.message_passing_rounds):
            if len(edges)>0:
                i_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
                p_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)

                # pattern receives sum of item embeddings
                p_msg = torch.zeros(m, self.hidden_dim)
                p_msg.index_add_(0, p_idx, h_item[i_idx])
                p_msg = F.relu(self.item_to_pattern(p_msg))
                h_pat = F.relu(h_pat + p_msg)

                # item receives sum of pattern embeddings
                i_msg = torch.zeros(n, self.hidden_dim)
                i_msg.index_add_(0, i_idx, h_pat[p_idx])
                i_msg = F.relu(self.pattern_to_item(i_msg))
                h_item = F.relu(h_item + i_msg)
            else:
                # no edges => skip
                pass

        # The GNN yields final item embeddings h_item
        # We do not do pair scoring here unless we want to handle it.
        return h_item, h_pat

class ItemPairBranchModel(nn.Module):
    """
    A wrapper that uses BranchingGNN to get item embeddings, 
    then for a given list of candidate pairs, produces a score for each pair.
    """
    def __init__(self, gnn: BranchingGNN):
        super().__init__()
        self.gnn = gnn
        # we define a small MLP for pair scoring
        self.pair_scorer = nn.Sequential(
            nn.Linear(2*gnn.hidden_dim, gnn.hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn.hidden_dim, 1)
        )

    def forward(self, gnn_input, candidate_pairs):
        """
        gnn_input: dict with item_feat, pattern_feat, edge_index_item_to_pattern
        candidate_pairs: list of (i,j)
        
        Returns a 1D tensor of shape [len(candidate_pairs)] with scores for each pair
        """
        h_item, h_pat = self.gnn(gnn_input)
        # compute pair embeddings
        scores_list = []
        for (i,j) in candidate_pairs:
            # concat h_item[i], h_item[j]
            eij = torch.cat([h_item[i], h_item[j]], dim=0)
            s = self.pair_scorer(eij.unsqueeze(0))  # shape [1,1]
            scores_list.append(s)
        scores = torch.cat(scores_list, dim=0).squeeze(-1)  # shape [num_pairs]
        return scores
