import numpy as np

def prob_region_inclusion(s, region_assignment, region):
    assert region in region_assignment, "Region not in assigned regions"

    region_probs = s[np.where(region_assignment == region)[0]]
    return np.max(region_probs)

def expected_num_new_regions(s, region_assigment)
    expected_num = 0
    for region in region_assignment:
        expected_num += prob_region_inclusion(s, region_assignment, region)

    return expected_num