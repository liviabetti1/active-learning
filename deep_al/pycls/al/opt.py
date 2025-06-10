import os
import dill
import numpy as np
import pandas as pd
import numpy as np

import cvxpy as cp
import mosek

from .cvxpy_fns import cost, util
from . import cost as np_cost

UTILITY_FNS = {
    "random": util.random,
    "greedycost": util.greedy,
    "poprisk": util.pop_risk
}

COST_FNS = {
    "uniform": cost.uniform,
    "pointwise_by_region": cost.pointwise_by_region,
    "pointwise_by_array": cost.pointwise_by_array,
    "clustered_by_region": cost.clustered_by_region
}

NP_COST_FNS = {
    "uniform": np_cost.uniform,
    "pointwise_by_region": np_cost.pointwise_by_region,
    "pointwise_by_array": np_cost.pointwise_by_array,
    "clustered_by_region": np_cost.clustered_by_region
}

class Opt:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet
        self.uSet = uSet
        self.budget = budgetSize
        self._resolve_cost_func()

    def set_utility_func(self, utility_func_type):
        if utility_func_type not in UTILITY_FNS:
            raise ValueError(f"Invalid utility function type: {utility_func_type}")
        self.utility_func_type = utility_func_type
        self.utility_func = UTILITY_FNS[utility_func_type]

        if utility_func_type == "poprisk":
            assert self.cfg.GROUPS.GROUP_ASSIGNMENT is not None, "Group assignment must not be none for poprisk utility function"

            self.group_assignment = np.array(self.cfg.GROUPS.GROUP_ASSIGNMENT)

            relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
            self.group_assignment = self.group_assignment[relevant_indices]

            self.utility_func = lambda s: util.pop_risk(s, self.group_assignment)
        print(f"Utility function set to: {utility_func_type}")

    def _resolve_cost_func(self):
        cost_func_type = self.cfg.COST.FN
        self.cost_func_type = cost_func_type
        assert cost_func_type in COST_FNS, f"Invalid cost function type: {cost_func_type}"

        self.cost_func = COST_FNS[cost_func_type]
        self.np_cost_func = NP_COST_FNS[cost_func_type]

        if cost_func_type == "pointwise_by_region" or "clustered_by_region":
            assert self.cfg.COST.REGION_ASSIGNMENT is not None, \
                "Region Assignment must not be None for 'pointwise_by_region' cost function"
            self.region_assignment = np.array(self.cfg.COST.REGION_ASSIGNMENT)

            relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
            labeled_inclusion_vector = np.concatenate([np.ones(len(self.lSet)), np.zeros(len(self.uSet))]).astype(bool)

            self.region_assignment = self.region_assignment[relevant_indices] #need to align region assignment with indices

            if cost_func_type == "pointwise_by_region":
                self.cost_func = lambda s: cost.pointwise_by_region(s, self.region_assignment, labeled_inclusion_vector, labeled_region_cost=self.cfg.COST.LABELED_REGION_COST, new_region_cost=self.cfg.COST.NEW_REGION_COST)
                self.np_cost_func = lambda s: np_cost.pointwise_by_region(s, self.region_assignment, labeled_inclusion_vector, labeled_region_cost=self.cfg.COST.LABELED_REGION_COST, new_region_cost=self.cfg.COST.NEW_REGION_COST)
            elif cost_func_type == "clustered_by_region":
                self.cost_func = lambda s: cost.clustered_by_region(s, self.region_assignment, labeled_inclusion_vector, labeled_region_cost=self.cfg.COST.LABELED_REGION_COST, new_region_cost=self.cfg.COST.NEW_REGION_COST)
                self.np_cost_func = lambda s: np_cost.clustered_by_region(s, self.region_assignment, labeled_inclusion_vector, labeled_region_cost=self.cfg.COST.LABELED_REGION_COST, new_region_cost=self.cfg.COST.NEW_REGION_COST)

        elif cost_func_type == "pointwise_by_array":
            assert self.cfg.COST.ARRAY is not None, "Cost array must not be None for this cost function"
            self.cost_array = self.cfg.COST.ARRAY

            self.cost_func = lambda s: cost.pointwise_by_array(s, self.cost_array)
            self.np_cost_func = lambda s: np_cost.pointwise_by_array(s, self.cost_array)

    def solve_opt(self):
        assert self.utility_func_type != "Random", "Please do not use the optimization function for random selection"

        relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        labeled_inclusion_vector = np.concatenate([np.ones(len(self.lSet)), np.zeros(len(self.uSet))]).astype(bool)

        n = len(relevant_indices)
        s = cp.Variable(n, nonneg=True)

        assert s.shape == (len(relevant_indices),), f"s should be shape {(len(relevant_indices),)}, got {s.shape}"
        assert labeled_inclusion_vector.shape == (len(relevant_indices),), f"labeled_inclusion_vector should be shape {(len(relevant_indices),)}, got {labeled_inclusion_vector.shape}"

        objective = self.utility_func(s)
        constraints = [
            0 <= s,
            s <= 1,
            self.cost_func(s) <= self.budget + len(self.lSet), #budget only accounts for additional points
            s[labeled_inclusion_vector] == 1
        ]
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver = cp.MOSEK)

        assert prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], \
            f"Optimization failed. Status: {prob.status}"

        if prob.status == cp.OPTIMAL_INACCURATE:
            print("Warning: Solution is OPTIMAL_INACCURATE. Results may be unreliable.")

        print("Optimal s is: ", s.value)

        return s.value

    def select_samples(self):
        # using only labeled+unlabeled indices, without validation set.
        assert hasattr(self, "cost_func") and self.cost_func is not None, "Need to specify cost function"

        relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)

        all_indices = np.arange(len(relevant_indices))
        existing_indices = np.arange(len(self.lSet))
        non_existing_indices = np.setdiff1d(all_indices, existing_indices)

        sample_inclusion_vector = np.concatenate([np.ones(len(self.lSet)), np.zeros(len(self.uSet))])
        np.random.seed(self.seed)

        if self.utility_func_type == "random":
            permuted_indices = np.random.permutation(non_existing_indices)

            for i in range(len(permuted_indices)):
                sample_inclusion_vector[permuted_indices[i]] = 1
                if self.np_cost_func(sample_inclusion_vector) > self.budget + len(self.lSet):
                    sample_inclusion_vector[permuted_indices[i]] = 0
                    break
        else:
            assert hasattr(self, "utility_func") and self.utility_func is not None, "Need to specify utility function"

            probs = self.solve_opt()
            for i in range(len(relevant_indices)):
                draw = np.random.choice([0,1], p=[1-probs[i], probs[i]])
                sample_inclusion_vector[i] = draw
        
        selected = np.where(sample_inclusion_vector == 1)[0][len(self.lSet):]

        total_sample_cost = self.np_cost_func(sample_inclusion_vector)
        print(f"Total Sample Cost: {total_sample_cost}")

        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        if self.utility_func_type == "random":
            return activeSet, remainSet, total_sample_cost
        else:
            return activeSet, remainSet, total_sample_cost, probs, relevant_indices