# evaluation/metrics.py

import csv
import statistics

def summarize_csv_performance(csv_file):
    """
    Reads 'strategy_comparison.csv' and computes average time, nodes, obj per strategy.
    """
    data = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["time"] = float(row["time"])
            row["nodes"] = int(row["nodes"])
            row["best_obj"] = float(row["best_obj"])
            row["instance_id"] = int(row["instance_id"])
            data.append(row)
    # group by strategy
    strategies = {}
    for row in data:
        s = row["strategy"]
        if s not in strategies:
            strategies[s] = []
        strategies[s].append(row)
    # compute stats
    results = {}
    for s, rows in strategies.items():
        avg_time = statistics.mean(r["time"] for r in rows)
        avg_nodes = statistics.mean(r["nodes"] for r in rows)
        avg_obj = statistics.mean(r["best_obj"] for r in rows)
        results[s] = {"avg_time":avg_time, "avg_nodes":avg_nodes, "avg_obj":avg_obj}
    return results
