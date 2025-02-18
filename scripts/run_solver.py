# scripts/run_solver.py

import pickle
from solver.branch_and_price_solver import CSPBranchAndPriceSolver
from solver.branching_strategies import RyanFosterStrategy, NaiveBranchingStrategy

def main():
    # load single instance
    with open("single_instance.pkl","rb") as f:
        inst = pickle.load(f)
    # pick strategy
    strategy = RyanFosterStrategy()
    solver = CSPBranchAndPriceSolver(inst["roll_length"], inst["item_lengths"], inst["demands"], strategy)
    best_obj, best_sol = solver.solve()
    print(f"Best objective = {best_obj}")
    print("Solution usage:")
    patterns, x_vals = best_sol
    for j, x in enumerate(x_vals):
        if x>1e-6:
            print(f"Pattern {j}, usage={x}, pattern={patterns[j]}")

if __name__=="__main__":
    main()
