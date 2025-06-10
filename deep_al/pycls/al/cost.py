import numpy as np
import cvxpy as cp

def uniform(s):
    return np.sum(s)

def pointwise_by_array(s, cost_array):
    assert len(s) == len(cost_array), "Cost array length mismatch"
    return s @ cost_array

def pointwise_by_region(s, region_assignment, labeled_inclusion_vector, labeled_region_cost=1, new_region_cost=2):
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

    # Create costs array
    costs = np.array([
        labeled_region_cost if region_assignment[i] in labeled_regions else new_region_cost
        for i in range(n)
    ])

    return s @ costs

def clustered_by_region(s, region_assignment, labeled_inclusion_vector, labeled_region_cost=1, new_region_cost=2):
    '''
    Assigns cost of labeled_region_cost if the sample is in a region that already has a labeled sample,
    and new_region_cost otherwise.

    Inputs:
        s: cvxpy Variable of shape (n,)
        region_assignment: 1D numpy array of length n
        labeled_inclusion_vector: boolean numpy array of length n
        labeled_region_cost: scalar
        new_region_cost: scalar

    Output:
        CVXPY expression for expected cost: s @ costs
    '''
    n = s.shape[0]
    assert len(region_assignment) == n, "Region assignment length mismatch"
    assert len(labeled_inclusion_vector) == n, "Labeled inclusion vector length mismatch"

    labeled_regions = np.unique(region_assignment[labeled_inclusion_vector])
    all_regions = np.unique(region_assignment)

    expected_new_region_cost = 0.0

    for region in all_regions:
        #if region is already labeled, no new region cost
        if region in labeled_regions:
            region_cost = 0
            continue
        else:
            region_cost = new_region_cost

        region_points = np.where(region_assignment == region)[0]
        s_region = s[region_points]
        
        #probability region not selected = product over points of (1 - s_i)
        p_not_selected = np.prod(1 - s_region) #worried this might be unstable, depending on what s is
        p_selected = 1 - p_not_selected
        
        expected_new_region_cost += p_selected * region_cost
    
    #expected point collection cost: sum of s
    expected_point_cost = np.sum(s) * labeled_region_cost 
    
    #total expected cost
    return expected_new_region_cost + expected_point_cost

