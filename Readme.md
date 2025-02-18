# Solver and ML Package Overview

## Solver Package (`solver/`)

### `column_generation.py`
**Role:** Implements the pricing subproblem for the cutting stock problem—either as a knapsack or ILP approach—to generate new columns (patterns).

**What it does:**
- Defines a `ColumnGenerator` class that, given dual prices from the master LP, solves a knapsack-like subproblem to produce a new cutting pattern.
- Implements two methods for pricing:
  - **ILP approach** using Gurobi.
  - **DP approach** for the unbounded knapsack.

**How it fits:**
- The solver calls `column_generation.ColumnGenerator.generate_pattern(duals)` to discover new columns (cutting patterns) with negative reduced cost.
- These new columns are appended to the master problem to refine the LP solution.

### `branching_strategies.py`
**Role:** Implements branching strategies (naive, Ryan-Foster, GNN-based).

**What it does:**
- `BaseBranchingStrategy`: Abstract base class with `choose_branch` method.
- `NaiveBranchingStrategy`: Selects the first fractional pattern variable to branch on.
- `RyanFosterStrategy`: Chooses a pair of items `(i,j)` fractionally served together, then creates “together/apart” branches.
- `GNNBranchingStrategy`: Calls a trained GNN model to choose a branching pair based on solver state.

**How it fits:**
- The `CSPBranchAndPriceSolver` calls `branching_strategies.*.choose_branch(...)` to determine a branching decision.
- Swappable strategies allow easy switching between naive, Ryan-Foster, or ML-based branching.

### `node.py`
**Role:** Defines the `SearchNode` structure for the branch-and-bound tree.

**What it does:**
- `SearchNode`: A namedtuple storing patterns, branching constraints, node depth, and LP relaxation objective.
- `copy_node(...)`: Clones or partially deep-copies a node when creating child branches.

**How it fits:**
- Standardized data structure for passing node information as the solver descends the branch-and-bound tree.
- When branching, two child `SearchNode`s are created from the current node.

### `logger.py`
**Role:** Structured CSV logging of solver events.

**What it does:**
- `SolverLogger` opens a CSV file, writes a header, and logs events with timestamps, node depth, and detail text.
- Ensures logs are flushed to disk for tracking solver progress.

**How it fits:**
- Used for detailed logging of events such as “ColumnAdded”, “Branch”, “Prune”, etc.
- Logs can be aggregated later for performance analysis.

### `branch_and_price_solver.py`
**Role:** Core Branch-and-Price solver, orchestrating:
- Master LP solves (via Gurobi).
- Column generation (`column_generation.ColumnGenerator`).
- Branch-and-bound search.
- Branching strategies (`branching_strategies.py`).

**What it does:**
- `CSPBranchAndPriceSolver`:
  - `solve()`: Initializes a root node and calls `_branch_and_price_node`.
  - `_branch_and_price_node(node)`: Performs column generation, checks integrality, prunes if needed, or branches using a chosen strategy.
  - `_solve_node_lp(node)`: Repeatedly solves LP + does pricing until no new column is found.
  - `_solve_lp_relaxation(...)`: Builds and solves the master LP, extracts solutions and duals.

**How it fits:**
- Calls into `column_generation` for new columns.
- Relies on a branching strategy for decision-making.
- Optionally logs events via `logger.py`.
- Maintains the best integer solution found, pruning unpromising nodes.

## ML Package (`ml/`)

### `graph_utils.py`
**Role:** Converts solver state (item demands, patterns, LP solution) into a graph representation for a GNN.

**What it does:**
- `build_state_graph_item_pairs(...)`:
  - Creates item node features.
  - Creates pattern node features.
  - Builds an edge list for items in patterns.
  - Identifies candidate item pairs for Ryan-Foster-style branching.

**How it fits:**
- Used by `GNNBranchingStrategy` for inference.
- Useful for training dataset generation.

### `model.py`
**Role:** Defines the Graph Neural Network architectures for branching decisions.

**What it does:**
- `BranchingGNN`: Bipartite message-passing network updating item and pattern embeddings.
- `ItemPairBranchModel`: Uses `BranchingGNN` to score candidate branching pairs.

**How it fits:**
- `GNNBranchingStrategy` uses these models for branching decisions.
- Models are trained on expert branching decisions.

### `dataset.py`
**Role:** Defines PyTorch dataset loaders for GNN training.

**What it does:**
- `BranchingDataset`: Stores (gnn_input, label) pairs.
- `collate_fn`: Handles variable-size graphs in batches.

**How it fits:**
- Used in supervised training.
- Enables efficient batch processing for GNN training.

### `train_supervised.py`
**Role:** Implements supervised learning for the GNN.

**What it does:**
- `train_gnn_supervised(...)`:
  - Runs training epochs.
  - Computes cross-entropy loss against expert labels.
  - Logs loss to TensorBoard.
  - Saves trained model.

**How it fits:**
- Trains a model to mimic expert branching decisions.

### `train_rl.py`
**Role:** Implements reinforcement learning (REINFORCE) for GNN refinement.

**What it does:**
- `fine_tune_with_reinforcement(...)`: Runs RL-based optimization for fewer B&B nodes.
- Logs log-probs for policy gradient updates.

**How it fits:**
- Allows improving branching beyond expert imitation.

## Dependencies (`requirements.txt`)
- Lists required Python packages (Gurobi, PyTorch, etc.) for reproducibility.
