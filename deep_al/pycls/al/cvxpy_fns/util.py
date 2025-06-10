import cvxpy as cp

def random(s):
    return 0

def greedy(s):
    return cp.sum(s)

#Workshop paper objective (from rep. matters paper)
def pop_risk(s, groups, l=0.5):
    """
    Compute weighted risk over groups from input values and group labels,
    using group weights proportional to group frequency in `groups`.
    
    Args:
        x: cp.array of values (e.g., weights or indicators)
        groups: numpy array of group labels
        l: weighting parameter in [0,1]
        
    Returns:
        scalar risk value (cp scalar)
    """
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    group_weights = cp.array(group_counts / group_counts.sum())  #proportions
    
    group_sizes = cp.array([cp.sum(s[groups == g]) for g in unique_groups])
    total_size = cp.sum(group_sizes)
    
    # risk per group: l*(1/nj) + (1-l)*(1/n)
    group_risks = l / group_sizes + (1 - l) / total_size
    weighted_risks = group_weights * group_risks
    return -cp.sum(weighted_risks) #negative since opt will maximize this