import pytest
from bayesee.generation import *
import numpy as np


def test_euclidean_distance_exponential_typical_cases():
    size = (3,)
    center = (1,)
    exponent = -1
    output = euclidean_distance_exponential(size, center, exponent)
    assert output[center] == 0, f"Value at the center is not 0, but {output[center]}."
    assert output[0] == 1


def test_radial_asd_exponential_typical_cases():
    size = (3,)
    exponent = -1
    output = radial_asd_exponential(size, exponent)
    assert np.allclose(output.mean(), 0.0), f"Mean is not 0, but {output.mean()}."
    assert np.allclose(output.std(), 1.0), f"STD is not 1, but {output.std()}."


def test_sine_wave_typical_cases():
    size = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = sine_wave(size, frequencies)
    assert np.allclose(output[1, 2, 2], 0.0)
    assert len(np.unique(output[:, 0, 1])) == 3
    assert len(np.unique(output[0, 1, :])) == 1


def test_cosine_wave_typical_cases():
    size = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = cosine_wave(size, frequencies)
    assert len(np.unique(output[:, 0, 1])) == 2
    assert len(np.unique(output[0, 1, :])) == 1


def test_gaussian_2d_typical_cases():
    size = (4, 4)
    stds = 1
    assert np.allclose(gaussian_2d(size, stds)[2, 2], 1)

    stds = (1, 1)
    assert np.allclose(gaussian_2d(size, stds)[2, 2], 1)

    stds = (1, 1, 0.5)
    assert np.allclose(gaussian_2d(size, stds)[2, 2], 1)


def test_hann_typical_cases():
    size = (4, 4)
    center = (2, 2)
    width = 1
    assert np.allclose(hann(size, center, width)[2, 2], 1)


def test_square_typical_cases():
    size = (3,)
    center = (1,)

    widths = (1,)
    assert np.allclose(square(size, center, widths), [0, 1, 0])

    widths = (3,)
    assert np.allclose(square(size, center, widths), 1)


def test_circle_typical_cases():
    size = (3,)
    center = (1,)

    radii = (1,)
    assert np.allclose(circle(size, center, radii), [0, 1, 0])

    radii = (3,)
    assert np.allclose(circle(size, center, radii), 1)


def test_probability_sample_dots_edge_cases():
    x = np.ones((3, 3))
    x[0, 0] = 0
    output = probability_sample_dots(x)
    assert np.allclose(output[0, 0], 0)
    assert np.allclose(output[2, 2], 1)
