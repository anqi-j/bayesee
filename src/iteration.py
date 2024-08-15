import numpy as np
import pandas as pd


def bootstrap_indexes(array_index):
    return np.random.choice(array_index, size=len(array_index), replace=True)


def bootstrap_binned_indexes(array_index, array_bin):
    bootstrap_array_index = np.zeros_like(array_index)
    for index_bin in np.unique(array_bin):
        select_bin = array_bin == index_bin
        select_array_index = array_index[select_bin]
        bootstrap_array_index[select_bin] = bootstrap_indexes(select_array_index)

    return bootstrap_array_index
