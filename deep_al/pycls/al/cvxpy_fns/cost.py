import cvxpy as cp

def uniform(s):
    return cp.sum(s)

def pointwise_by_EA(s, cost_array):
    assert s.shape[0] == len(cost_array), "Cost array length mismatch"
    return s @ cost_array