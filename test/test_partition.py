import pytest
from bayesee.partition import *
import numpy as np
import pandas as pd


def test_independent_bin_typical_cases():
    df = pd.DataFrame(
        {
            "f1": np.arange(8),
            "f2": np.arange(8, 0, -1),
            "f3": np.arange(8),
        }
    )
    bin_columns = ["f1", "f2", "f3"]
    n_bins = 2

    df_binned = independent_bin(df, bin_columns, n_bins)

    assert np.allclose(df_binned.loc[0:3, "bin_f1"], 1)
    assert np.allclose(df_binned.loc[0:3, "bin_f2"], 2)
    assert np.allclose(df_binned.loc[0:3, "bin_f3"], 1)


def test_recursive_bin_typical_cases():
    df = pd.DataFrame(
        {
            "f1": np.arange(8),
            "f2": np.arange(8, 0, -1),
            "f3": np.arange(8),
        }
    )
    bin_columns = ["f1", "f2", "f3"]
    n_bins = [2, 2, 2]

    df_binned = recursive_bin(df, bin_columns, n_bins)

    assert (df_binned["level3_bin_f3"] == np.arange(1, 9)).all()
