import pytest
from bayesee.generation import *
import numpy as np


def test_euclidean_distance_exponential_typical_cases():
    shape = (3,)
    center = (1,)
    exponent = -1
    output = euclidean_distance_exponential(shape, center, exponent)
    assert output[center] == 0, f"Value at the center is not 0, but {output[center]}."
    assert output[0] == 1


def test_radial_asd_exponential_typical_cases():
    shape = (3,)
    exponent = -1
    output = radial_asd_exponential(shape, exponent)
    assert np.allclose(output.mean(), 0.0), f"Mean is not 0, but {output.mean()}."
    assert np.allclose(output.std(), 1.0), f"STD is not 1, but {output.std()}."


def test_orientation_angle_2d_typical_cases():
    shape = (1, 1)
    center = (0, 0)
    assert orientation_angle_2d(shape, center)[0][0] == 0

    shape = (3, 3)
    center = (1, 1)
    assert orientation_angle_2d(shape, center)[1, -1] == 0


def test_sine_wave_typical_cases():
    shape = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = sine_wave(shape, frequencies)
    assert np.allclose(output[1, 2, 2], 0.0)
    assert len(np.unique(output[:, 0, 1])) == 3
    assert len(np.unique(output[0, 1, :])) == 1


def test_cosine_wave_typical_cases():
    shape = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = cosine_wave(shape, frequencies)
    assert len(np.unique(output[:, 0, 1])) == 2
    assert len(np.unique(output[0, 1, :])) == 1


def test_sine_sawtooth_wave_typical_cases():
    shape = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = sine_sawtooth_wave(shape, frequencies)
    assert np.allclose(output[1, 2, 2], 0.0)
    assert len(np.unique(output[:, 0, 1])) == 3
    assert len(np.unique(output[0, 1, :])) == 1


def test_cosine_sawtooth_wave_typical_cases():
    shape = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = cosine_sawtooth_wave(shape, frequencies)
    assert len(np.unique(output[:, 0, 1])) == 3
    assert len(np.unique(output[0, 1, :])) == 1


def test_sine_square_wave_typical_cases():
    shape = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = sine_square_wave(shape, frequencies)
    assert len(np.unique(output[:, 0, 1])) == 2
    assert len(np.unique(output[0, 1, :])) == 1


def test_cosine_square_wave_typical_cases():
    shape = (3, 3, 3)
    frequencies = (1, 1, 0)
    output = cosine_square_wave(shape, frequencies)
    assert len(np.unique(output[:, 0, 1])) == 2
    assert len(np.unique(output[0, 1, :])) == 1


def test_gaussian_2d_typical_cases():
    shape = (4, 4)
    stds = 1
    assert np.allclose(gaussian_2d(shape, stds)[2, 2], 1)

    stds = (1, 1)
    assert np.allclose(gaussian_2d(shape, stds)[2, 2], 1)

    stds = (1, 1, 0.5)
    assert np.allclose(gaussian_2d(shape, stds)[2, 2], 1)


def test_hann_typical_cases():
    shape = (4, 4)
    center = (2, 2)
    width = 1
    assert np.allclose(hann(shape, center, width)[2, 2], 1)


def test_flat_top_hann_typical_cases():
    shape = (4, 4)
    center = (2, 2)
    head_width = 1
    foot_width = 2
    assert np.allclose(flat_top_hann(shape, center, head_width, foot_width)[2, 2], 1)


def test_square_typical_cases():
    shape = (3,)
    center = (1,)

    widths = (1,)
    assert np.allclose(square(shape, center, widths), [0, 1, 0])

    widths = (3,)
    assert np.allclose(square(shape, center, widths), 1)


def test_circle_typical_cases():
    shape = (3,)
    center = (1,)

    radii = (1,)
    assert np.allclose(circle(shape, center, radii), [0, 1, 0])

    radii = (3,)
    assert np.allclose(circle(shape, center, radii), 1)


def test_typical_grid_edge_cases():
    shape = (3, 3)
    origin = np.array((1, 1))
    unit_distance = 2
    angle_step = np.pi / 3
    assert np.array_equal(
        typical_grid([], shape, origin, unit_distance, angle_step)[0], origin
    )


def test_kernel_on_grid_edge_cases():
    shape = (2, 2)
    list_center = [np.array([1, 1])]
    kernel = np.ones((2, 2))
    assert np.allclose(kernel_on_grid(shape, list_center, kernel), 1)


def test_radial_grid_edge_cases():
    shape = (3, 3)
    origin = np.array((1, 1))
    unit_distance = 2
    angle_step = np.pi / 3
    assert np.array_equal(
        radial_grid(shape, origin, unit_distance, angle_step)[0], origin
    )


def test_probability_sample_dots_edge_cases():
    x = np.ones((3, 3))
    x[0, 0] = 0
    output = probability_sample_dots(x)
    assert np.allclose(output[0, 0], 0)
    assert np.allclose(output[2, 2], 1)
