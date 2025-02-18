# Cutting Stock Branch-and-Price with GNN Branching

## Overview
This project implements a large-scale cutting stock solver using Branch-and-Price in Python with Gurobi. It supports multiple branching strategies:
- Naive pattern branching
- Ryan-Foster item-pair branching
- GNN-based learned branching

We provide scripts for:
- Generating random instances
- Collecting expert data for supervised training
- Training a bipartite GNN
- Evaluating solver performance and classification accuracy

## Installation
1. Install Python >= 3.8
2. Install Gurobi and obtain a license
3. `pip install -r requirements.txt`

## Usage
1. `python scripts/generate_data.py` to generate random instances and collect training data from an expert.
2. `python scripts/train_gnn.py` to train the GNN on the collected data.
3. `python scripts/evaluate_supervised.py` to evaluate classification accuracy of the model on a test set.
4. `python scripts/compare_strategies.py` to run naive, Ryan-Foster, and GNN-based branching on test instances, logging results to CSV.
