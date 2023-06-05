import numpy as np
import pandas as pd


def bootstrap_indexes(array_index):
    return np.random.choice(array_index, size=len(array_index), replace=True)
