# ml/gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchingGNN(nn.Module):
    def __init__(self, var_feat_dim, constr_feat_dim, hidden_dim=64, message_passing_rounds=3):
        super(BranchingGNN, self).__init__()
        self.message_passing_rounds = message_passing_rounds
        # Linear layers for initial feature embedding
        self.var_embed = nn.Linear(var_feat_dim, hidden_dim)
        self.constr_embed = nn.Linear(constr_feat_dim, hidden_dim)
        # Message passing transforms
        self.msg_var_to_constr = nn.Linear(hidden_dim, hidden_dim)
        self.msg_constr_to_var = nn.Linear(hidden_dim, hidden_dim)
        # Output scoring layer
        self.score_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        state: dict with
          'var_feat' -> tensor [V, var_feat_dim]
          'constr_feat' -> tensor [C, constr_feat_dim]
          'edge_index_var_to_constr' -> list of (v, c) edges
        """
        var_feat = state['var_feat']
        constr_feat = state['constr_feat']
        edge_index = state['edge_index_var_to_constr']

        V = var_feat.size(0)
        C = constr_feat.size(0)

        # 1) Initial embeddings
        h_var = F.relu(self.var_embed(var_feat))         # shape [V, hidden_dim]
        h_constr = F.relu(self.constr_embed(constr_feat))# shape [C, hidden_dim]

        # 2) Message passing rounds
        for _ in range(self.message_passing_rounds):
            # Variables -> Constraints
            if edge_index:
                var_idx = torch.tensor([e[0] for e in edge_index], dtype=torch.long)
                constr_idx = torch.tensor([e[1] for e in edge_index], dtype=torch.long)
                messages_to_constr = torch.zeros(C, h_var.size(1))
                messages_to_constr.index_add_(0, constr_idx, h_var[var_idx])
            else:
                messages_to_constr = torch.zeros(C, h_var.size(1))

            h_constr = F.relu(h_constr + self.msg_var_to_constr(messages_to_constr))

            # Constraints -> Variables
            if edge_index:
                var_idx = torch.tensor([e[0] for e in edge_index], dtype=torch.long)
                constr_idx = torch.tensor([e[1] for e in edge_index], dtype=torch.long)
                messages_to_var = torch.zeros(V, h_constr.size(1))
                messages_to_var.index_add_(0, var_idx, h_constr[constr_idx])
            else:
                messages_to_var = torch.zeros(V, h_constr.size(1))

            h_var = F.relu(h_var + self.msg_constr_to_var(messages_to_var))

        # 3) Final scoring for each variable (pattern)
        scores = self.score_layer(h_var).view(-1)  # shape [V]
        return scores
