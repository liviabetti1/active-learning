import numpy as np

def uniform(s):
    return np.sum(s)

def pointwise_by_EA(s, cost_array):
    assert len(s) == len(cost_array), "Cost array length mismatch"
    return s @ cost_array