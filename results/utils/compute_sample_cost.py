import os
import numpy as np
from tqdm import tqdm
import glob
import yaml

import deep_al.pycls.al.cost as cost

COST_FNS = {
    "uniform": cost.uniform,
    "pointwise_by_array": cost.pointwise_by_array,
    "unit_aware_pointwise_cost": cost.unit_aware_pointwise_cost
}

def load_config_as_dict(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict

def resolve_cost_func(config_dict, lSet, uSet):
    relevant_indices = np.concatenate([lSet, uSet]).astype(int)

    unit_assignment = config_dict["UNITS"]["UNIT_ASSIGNMENT"][relevant_indices]

    cost_func_type = config_dict["COST"]["FN"]
    cost_func = COST_FNS[cost_func_type]
    cost_array = config_dict["COST"]["ARRAY"][relevant_indices]

    units = np.unique(unit_assignment)
    unit_cost = config_dict["COST"]["UNIT_COST"]

    if cost_func_type == "pointwise_by_array":
        if unit_cost is not None:
            cost_dict = unit_cost[0] #since is an array
            cost_array = np.array([cost_dict[u] for u in units])
        elif cost_array is not None:
            cost_array = cost_array[relevant_indices]
        else:
            raise(AssertionError)

        cost_func = lambda s: cost.pointwise_by_array(s, cost_array)c

    elif cost_func_type == "unit_aware_pointwise_cost":
        labeled_indices = np.array([int(idx in set(lSet)) for idx in relevant_indices])

        units = np.arange(len(relevant_indices)) #don't need for these purposes I don't think

        labeled_units = set(unit_assignment[lSet])
        unit_labeled_array = [unit_assignment[i] in labeled_units for i in range(len(relevant_indices))]

        cost_func = lambda s: cost.unit_aware_pointwise_cost(s, labeled_indices, unit_labeled_array)

    return cost_func

def assign_cost(
    active_set_path,
    labeled_set_path,
    unlabeled_set_path,
    config_dict,
    budget
):
    """
    Processes one activeSet.npy file, truncates based on cost, and saves the result.
    """
    activeSet = np.load(active_set_path)
    lSet = np.load(labeled_set_path)
    uSet = np.load(unlabeled_set_path)
    relevant_indices = np.concatenate([lSet, uSet]).astype(int)

    cost_func = resolve_cost_func(config_dict, lSet, uSet)

    unit_inclusion_vector = np.array([int(idx in set(lSet)) for idx in relevant_indices])
    total_cost = cost_func(unit_inclusion_vector)
    truncated_indices = []

    for idx in active_set:
        trial_unit_inclusion_vector = unit_inclusion_vector.copy()
        idx_pos = np.where(relevant_indices == idx)[0]
        trial_unit_inclusion_vector[idx_pos] = 1

        trial_total_cost = cost_func(trial_unit_inclusion_vector)
        if trial_total_cost > total_cost + budget:
            break

        truncated_indices.append(idx)
        unit_inclusion_vector = trial_unit_inclusion_vector
        total_cost = trial_total_cost

    return np.array(truncated_indices), total_cost
