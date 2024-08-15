import pytest
from bayesee.iteration import *
import numpy as np
import pandas as pd


def test_bootstrap_indexes_edge_cases():
    x = np.ones((3,))
    output = bootstrap_indexes(x)
    assert np.allclose(output, 1)


def test_bootstrap_binned_indexes_edge_cases():
    array_index = np.arange(4)
    array_bin = np.arange(4)
    assert (bootstrap_binned_indexes(array_index, array_bin) == array_index).all()
