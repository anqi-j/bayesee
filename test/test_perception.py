import pytest
from bayesee.perception import *
import numpy as np


def test_filter_fft_edge_cases():
    x = np.random.rand(3, 3, 3)

    fft_filter = np.ones_like(x)
    assert np.allclose(x, filter_fft(x, fft_filter))

    fft_filter = np.zeros_like(x)
    assert np.allclose(fft_filter, filter_fft(x, fft_filter))


def test_cosine_similarity_edge_cases():
    x = np.ones((3, 3))
    y = np.ones((3, 3)) * 3
    assert np.allclose(cosine_similarity(x, y), 1)

    x = np.sin(np.linspace(0, 2 * np.pi, 100))
    y = np.cos(np.linspace(0, 2 * np.pi, 100))
    assert np.allclose(cosine_similarity(x, y), 0)


def test_fft_cosine_similarity_edge_cases():
    x = np.ones((3, 3))
    y = np.ones((3, 3)) * 3
    assert np.allclose(fft_cosine_similarity(x, y), 1)


def test_abs_cosine_similarity_edge_cases():
    x = np.ones((3, 3))
    y = np.ones((3, 3)) * 3
    assert np.allclose(abs_cosine_similarity(x, y), 1)


def test_angle_cosine_similarity_edge_cases():
    x = np.ones((3, 3))
    x[:, 0] = -1
    y = x.copy()
    assert np.allclose(angle_cosine_similarity(x, y), 1)

def test_rmse_edge_cases():
    x = np.ones((3, 3))
    assert np.allclose(rmse(x, x), 0)
    
    y = np.zeros((3, 3))
    assert np.allclose(rmse(x, y), 1)
    
def test_mae_edge_cases():
    x = np.ones((3, 3))
    assert np.allclose(mae(x, x), 0)
    
    y = np.zeros((3, 3))
    assert np.allclose(mae(x, y), 1)

def test_pearson_edge_cases():
    x = np.identity(3)
    assert np.allclose(pearson(x, x), 1)
    assert np.allclose(pearson(x, -x), -1)
    
def test_ssim_edge_cases():
    x = np.identity(7)
    assert np.allclose(ssim(x, x), 1)

def test_normalize_min_max_typical_cases():
    x = np.random.rand(3, 3, 3)
    assert np.allclose(normalize_min_max(x).min(), 0)
    assert np.allclose(normalize_min_max(x).max(), 1)
    assert np.allclose(normalize_min_max(x, 1, 3).min(), 1)
    assert np.allclose(normalize_min_max(x, 1, 3).max(), 3)


def test_normalize_mean_std_typical_cases():
    x = np.random.rand(3, 3, 3)
    output = normalize_mean_std(x, 2, 4)
    assert np.allclose(output.mean(), 2)
    assert np.allclose(output.std(), 4)


def test_normalize_energy_typical_cases():
    x = np.random.rand(3, 3, 3)
    output = normalize_energy(x, 3)
    assert np.allclose(np.dot(output.flatten(), output.flatten()), 3)


def test_gamma_compress_typical_cases():
    bit = 8
    gamma = 2

    x = np.zeros((3, 3, 3))
    output = gamma_compress(x, 8, 2)
    assert (output == x).all()

    x = np.ones((3, 3, 3)) * (2**bit - 1)
    output = gamma_compress(x, 8, 2)
    assert (output == x).all()


def test_gamma_expand_typical_cases():
    bit = 8
    gamma = 2

    x = np.zeros((3, 3, 3))
    output = gamma_expand(x, 8, 2)
    assert (output == x).all()

    x = np.ones((3, 3, 3)) * (2**bit - 1)
    output = gamma_expand(x, 8, 2)
    assert (output == x).all()


def test_discrete_value_mapping_edge_cases():
    source = np.ones((3, 3))
    ixy = np.zeros((3, 3))
    jxy = np.zeros((3, 3))
    mapping = (ixy, jxy)
    shape = (2, 2)

    target, count, minima = discrete_value_mapping(source, mapping, shape)

    assert np.allclose(target, np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert np.allclose(count, np.array([[9, 0], [0, 0]]))
    assert np.allclose(minima, 0)


def test_human_contrast_sensitivity_function_typical_cases():
    shape = (4, 4)
    ppd = 1
    output_x = human_contrast_sensitivity_function(shape, ppd)
    assert output_x[2, 2] == 0

    output_y = human_contrast_sensitivity_function(shape, ppd * 2)
    assert np.allclose(output_x[0, 2], output_y[1, 2])
    assert np.allclose(output_x[2, 0], output_y[2, 1])
