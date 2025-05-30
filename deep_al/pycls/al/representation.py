import os
import dill
import numpy as np
import pandas as pd
import numpy as np

class Representation:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet
        self.uSet = uSet
        self.budget = budgetSize
        self._set_group_assignments()

    def _set_group_assignments(self):
        self.groups = np.array(self.cfg.GROUPS.GROUP_ASSIGNMENT)

    def select_samples(self):
        np.random.seed(self.seed)

        lSet = set(self.lSet)
        uSet = set(self.uSet)
        relevant_indices = np.concatenate([list(lSet), list(uSet)])
        all_groups = self.groups[relevant_indices]

        # Compute full group counts (denominator)
        full_group_labels, full_group_counts = np.unique(all_groups, return_counts=True)
        group_total = dict(zip(full_group_labels, full_group_counts))  # renamed for clarity

        selected = []
        for i in range(self.budget):
            labeled_groups = self.groups[list(lSet)]
            labeled_group_labels, labeled_group_counts = np.unique(labeled_groups, return_counts=True)
            group_labeled_dict = dict(zip(labeled_group_labels, labeled_group_counts))

            # Compute group representation ratio
            group_ratios = {}
            for group, total_count in group_total.items():
                labeled_count = group_labeled_dict.get(group, 0)
                group_ratios[group] = labeled_count / total_count

            # Sort groups by increasing representation ratio
            sorted_groups = sorted(group_ratios.items(), key=lambda x: x[1])

            chosen = None
            for group, ratio in sorted_groups:
                candidates = [idx for idx in uSet if self.groups[idx] == group]
                if candidates:
                    chosen = np.random.choice(candidates)
                    print(f"[{i+1}/{self.budget}] Selected index {chosen} from underrepresented group {group} (ratio: {ratio:.2f})")
                    break

            assert chosen is not None, (
                f"Selection failed at step {i+1}: uSet empty or no candidates found for any group."
            )

            # Update sets
            selected.append(chosen)
            uSet.remove(chosen)
            lSet.add(chosen)

        selected = np.array(selected)
        assert len(selected) == self.budget, "Should select exactly `budget` number of samples"
        assert len(np.intersect1d(selected, self.lSet)) == 0, "Selected samples should not already be in the labeled set"
        assert len(set(selected)) == len(selected), "Selected samples should be unique"

        activeSet = selected
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f"\nFinished the selection of {len(activeSet)} samples.")
        print(f"Active set: {activeSet}")
        print(f"Remaining unlabeled set size: {len(remainSet)}")
        return activeSet, remainSet