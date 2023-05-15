import pytest
from bayesee.interaction import *
import numpy as np

def test_filter_fft_edge_cases():
    x = np.random.rand(3,3,3)

    fft_filter = np.ones_like(x)
    assert np.allclose(x, filter_fft(x, fft_filter))

    fft_filter = np.zeros_like(x)
    assert np.allclose(fft_filter, filter_fft(x, fft_filter))
