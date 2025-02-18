# solver/logger.py

import csv
import os
import time

class SolverLogger:
    """
    Logs solver events (column generation, branching, node expansions) to CSV.
    Also handles timing.
    """
    def __init__(self, log_file="solver_log.csv"):
        self.log_file = log_file
        self.file_handle = None
        self.csv_writer = None
        self.start_time = time.time()

    def open(self):
        self.file_handle = open(self.log_file, "w", newline="")
        self.csv_writer = csv.writer(self.file_handle)
        # write header
        self.csv_writer.writerow(["timestamp","event","node_depth","lp_obj","details"])

    def close(self):
        if self.file_handle:
            self.file_handle.close()

    def log_event(self, event, node_depth, lp_obj, details=""):
        if not self.csv_writer:
            return
        t = time.time() - self.start_time
        self.csv_writer.writerow([f"{t:.2f}", event, node_depth, lp_obj, details])
        self.file_handle.flush()

    def __del__(self):
        self.close()
