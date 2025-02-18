# solver/node.py

from collections import namedtuple

SearchNode = namedtuple("SearchNode", ["patterns", "constraints", "depth", "lp_obj"])
"""
patterns: list of patterns (each pattern[i] is count of item i)
constraints: list of branching constraints, e.g. 
   [ {"type":"pair", "items":(i,j), "together":True}, 
     {"type":"pattern", "pattern_index":p, "must_use":False}, ...]
depth: int, the depth in the search tree
lp_obj: float, best LP relaxation objective found at this node
"""

def copy_node(node):
    """
    Create a shallow copy of the node, copying patterns if needed.
    constraints can be deep-copied if we plan to mutate them.
    """
    import copy
    return SearchNode(
        patterns = [list(p) for p in node.patterns],
        constraints = copy.deepcopy(node.constraints),
        depth = node.depth,
        lp_obj = node.lp_obj
    )
