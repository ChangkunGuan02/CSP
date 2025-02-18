# ml/graph_utils.py

import torch

def build_state_graph_item_pairs(solver, lp_solution, patterns):
    """
    Construct a bipartite graph representation:
      - item nodes: [n, item_feat_dim]
      - pattern nodes: [m, pattern_feat_dim]
      - edges: (item_index, pattern_index) if pattern includes item
    Then we identify candidate item pairs that appear fractionally together.

    Returns:
      gnn_input: a dict with 
        'item_feat' -> [n, dim] torch.Tensor
        'pattern_feat' -> [m, dim] torch.Tensor
        'edge_index_item_to_pattern' -> list of (i, p)
        'candidate_pairs' -> list of (i, j)
      The GNN can produce a score for each candidate pair. 
    """
    n = solver.n
    item_lengths = solver.item_lengths
    demands = solver.demands

    # item features: [demand_i, item_length_i, ...]
    item_feat = []
    for i in range(n):
        item_feat.append([demands[i], item_lengths[i]])
    item_feat_t = torch.tensor(item_feat, dtype=torch.float)

    # pattern features: [usage_fraction, #distinct_items, waste? ...]
    m = len(patterns)
    pattern_feat = []
    for p_idx, pat in enumerate(patterns):
        x_val = lp_solution[p_idx]
        distinct_items = sum(1 for c in pat if c>0)
        used_length = sum(pat[i]*item_lengths[i] for i in range(n))
        waste = solver.L - used_length
        pattern_feat.append([x_val, float(distinct_items), float(waste)])
    pattern_feat_t = torch.tensor(pattern_feat, dtype=torch.float)

    # build edges
    edges_item_to_pattern = []
    for p_idx, pat in enumerate(patterns):
        for i in range(n):
            if pat[i] > 0:
                edges_item_to_pattern.append((i, p_idx))

    # build candidate item pairs from fractional usage
    # i.e. if item i and j appear in some pattern p with x_val>0
    # for Ryan-Foster style
    alpha = {}
    for p_idx, pat in enumerate(patterns):
        x_val = lp_solution[p_idx]
        if x_val<1e-6:
            continue
        # gather items in this pattern
        items_in_pat = []
        for i in range(n):
            if pat[i]>0:
                items_in_pat.append(i)
        # pairwise combination
        for idx_a in range(len(items_in_pat)):
            for idx_b in range(idx_a+1, len(items_in_pat)):
                i = items_in_pat[idx_a]
                j = items_in_pat[idx_b]
                if i>j: i,j = j,i
                alpha[(i,j)] = alpha.get((i,j),0.0) + x_val
    # now filter out pairs with alpha in (0,1)
    # though we can list them all and let the GNN pick. 
    candidate_pairs = []
    for (i,j), val in alpha.items():
        if val>1e-6 and val<1-1e-6:
            candidate_pairs.append((i,j))

    gnn_input = {
        "item_feat": item_feat_t,
        "pattern_feat": pattern_feat_t,
        "edge_index_item_to_pattern": edges_item_to_pattern,
    }
    return gnn_input, candidate_pairs
