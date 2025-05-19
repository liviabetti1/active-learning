import sys
import os

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_al.pycls.datasets.usavars import USAVars


def get_ids(label, np_file):
    usavars = USAVars(root='/share/usavars', isTrain=True, label=label)