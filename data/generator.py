# data/generator.py

import random
import torch

def generate_random_instance(num_items, roll_length, min_item_length=1, max_item_length=None, 
                             min_demand=1, max_demand=10, seed=None):
    if seed is not None:
        random.seed(seed)
    if max_item_length is None:
        max_item_length = roll_length
    item_lengths = []
    demands = []
    for i in range(num_items):
        length = random.randint(min_item_length, max_item_length)
        # ensure length <= roll_length
        length = min(length, roll_length)
        dem = random.randint(min_demand, max_demand)
        item_lengths.append(length)
        demands.append(dem)
    instance = {
        "roll_length": roll_length,
        "item_lengths": item_lengths,
        "demands": demands
    }
    return instance

def generate_multiple_instances(count=10, **kwargs):
    instances = []
    for i in range(count):
        inst = generate_random_instance(**kwargs)
        instances.append(inst)
    return instances
