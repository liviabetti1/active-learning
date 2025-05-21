import sys
import os
import dill
import numpy as np

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_al.pycls.datasets.usavars import USAVars


def get_ids(label, np_file):
    usavars = USAVars(root='/share/usavars', isTrain=True, label=label)

    indices = np.load(np_file, allow_pickle=True)

    def get_usavars_ids(idx):
        _, _, _, id = usavars[idx]
        return id

    ids = [get_usavars_ids(idx) for idx in indices]

    return ids

def compute_total_sample_cost(np_file, label, cost_dict_path):
    ids = get_ids(label, np_file)

    with open(cost_dict_path, 'rb') as f:
        cost_dict = dill.load(f)

    total_cost = sum(cost_dict[id] for id in ids if id in cost_dict)

    return total_cost