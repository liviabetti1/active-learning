import cvxpy as cp
import numpy as np

def random(s):
    return 0

def greedy(s):
    return cp.sum(s)

def stratified(s, groups):
    unique_groups = np.unique(groups)
    
    group_sizes = cp.array([cp.sum(s[groups == g]) for g in unique_groups])
    
    # risk per group: l*(1/sqrt(nj)) + (1-l)*(1/sqrt(n))
    group_risks = l / cp.sqrt(group_sizes) + (1 - l) / cp.sqrt(total_size) #not weighted like next objective
    return -cp.sum(group_risks) #negative since opt will maximize this

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
    print(f"Population risk utility function with lambda={l}")
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    group_weights = cp.Constant(group_counts / group_counts.sum())  #proportions
    
    group_sizes = cp.hstack([cp.sum(s[groups == g]) for g in unique_groups])
    total_size = cp.sum(group_sizes)
    
    # risk per group: l*(1/sqrt(nj)) + (1-l)*(1/sqrt(n))
    group_risks = l * cp.inv_pos(cp.sqrt(group_sizes)) + (1 - l) * cp.inv_pos(cp.sqrt(total_size))
    weighted_risks = cp.multiply(group_weights, group_risks)
    return -cp.sum(weighted_risks) #negative since opt will maximize this

def similarity(s, similarity_matrix):
    test_similarity = similarity_matrix.sum(axis=1) #this sums the similarity for now, might want to change to softmax
    return s @ test_similarity

def diversity(s, distance_matrix):
    #slow as is, might need sparse distance implementation like knn
    return s @ distance_matrix @ s #penalizes close together points