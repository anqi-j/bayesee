import pytest
from bayesee.generation import *
import numpy as np

def test_euclidean_distance_exponential_typical_cases():
    size = (3,3,3)
    center = (1,2,1)
    exponent = -1
    output = euclidean_distance_exponential(size, center, exponent)
    assert output[center] == 0, f'Value at the center is not 0, but {output[center]}.'

def test_radial_asd_exponential_typical_cases():
    size = (3,3,3)
    exponent = -1
    output = radial_asd_exponential(size, exponent)
    assert np.allclose(output.mean(), 0.0), f'Mean is not 0, but {output.mean()}.'
    assert np.allclose(output.std(), 1.0), f'STD is not 1, but {output.std()}.'

    mean = 3
    std = 3
    output = radial_asd_exponential(size, exponent, mean, std)
    assert np.allclose(output.mean(), 3.0), f'Mean is not 3, but {output.mean()}.'
    assert np.allclose(output.std(), 3.0), f'STD is not 3, but {output.std()}.'
