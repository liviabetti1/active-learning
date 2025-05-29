import numpy as np
import cvxpy as cp

def uniform(s):
    return np.sum(s)

def pointwise_by_array(s, cost_array):
    assert len(s) == len(cost_array), "Cost array length mismatch"
    return s @ cost_array

def pointwise_by_region(s, region_assignment, labeled_inclusion_vector, **kwargs):
    '''
    Assigns cost of labeled_region_cost if the sample is in a region that already has a labeled sample,
    and new_region_cost otherwise.

    Inputs:
        s: array of inclusion of shape (n,)
        region_assignment: 1D numpy array of length n
        labeled_inclusion_vector: boolean numpy array of length n
        labeled_region_cost: scalar
        new_region_cost: scalar

    Output:
        expected cost s @costs
    '''
    n = len(s)
    assert len(region_assignment) == n, "Region assignment length mismatch"
    assert len(labeled_inclusion_vector) == n, "Labeled inclusion vector length mismatch"

    labeled_regions = np.unique(region_assignment[labeled_inclusion_vector])

    labeled_region_cost = kwargs.get('labeled_region_cost', 1)
    new_region_cost = kwargs.get('new_region_cost', 2)

    # Create costs array
    costs = np.array([
        labeled_region_cost if region_assignment[i] in labeled_regions else new_region_cost
        for i in range(n)
    ])

    return s @ costs