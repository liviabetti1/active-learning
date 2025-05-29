import numpy as np
import cvxpy as cp

def prob_region_inclusion(s, region_assignment, region):
    indices = np.where(region_assignment == region)[0]
    if len(indices) == 0:
        return 0  # region not found, skip

    return cp.maximum(*[s[i] for i in indices])

def expected_num_new_regions(s, region_assignment)
    expected_num = 0
    for region in np.unique(region_assignment):
        expected_num += prob_region_inclusion(s, region_assignment, region)

    return expected_num