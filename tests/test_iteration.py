import pytest
from bayesee.iteration import *
import numpy as np
import pandas as pd


def test_bootstrap_indexes_edge_cases():
    x = np.ones((3,))
    output = bootstrap_indexes(x)
    assert np.allclose(output, 1)
