import pytest
from bayesee.interaction import *
import numpy as np

def test_filter_fft_edge_cases():
    x = np.random.rand(3,3,3)

    fft_filter = np.ones_like(x)
    assert np.allclose(x, filter_fft(x, fft_filter))

    fft_filter = np.zeros_like(x)
    assert np.allclose(fft_filter, filter_fft(x, fft_filter))

def test_normalize_mean_std_typical_cases():
    x = np.random.rand(3,3,3)
    output = normalize_mean_std(x, 2, 4)
    assert np.allclose(output.mean(), 2)
    assert np.allclose(output.std(), 4)

def test_gamma_compress_typical_cases():
    bit = 8
    gamma = 2

    x = np.zeros((3,3,3))
    output = gamma_compress(x, 8, 2)
    assert (output == x).all()

    x = np.ones((3,3,3)) * (2**bit-1)
    output = gamma_compress(x, 8, 2)
    assert (output == x).all()

def test_gamma_expand_typical_cases():
    bit = 8
    gamma = 2

    x = np.zeros((3,3,3))
    output = gamma_expand(x, 8, 2)
    assert (output == x).all()

    x = np.ones((3,3,3)) * (2**bit-1)
    output = gamma_expand(x, 8, 2)
    assert (output == x).all()

def test_human_contrast_sensitivity_function_typical_cases():
    size = (4,4)
    ppd = 1
    output_x = human_contrast_sensitivity_function(size, ppd)
    assert output_x[2,2] == 0

    output_y = human_contrast_sensitivity_function(size, ppd*2)
    assert np.allclose(output_x[0,2], output_y[1,2])
    assert np.allclose(output_x[2,0], output_y[2,1])
