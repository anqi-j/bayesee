import pytest
from bayesee.evaluation import *
import numpy as np
import pandas as pd


def test_compute_dprime_criterion_w_discrete_response_typical_cases():
    stimulus = {
        "df": pd.DataFrame({"amplitude": np.ones((4,)), "presence": np.ones((4,))})
    }
    response = {"df": pd.DataFrame({"presence": np.zeros((4,))})}
    stimulus["df"]["presence"][0:2] = 0

    array_dprime, array_criterion = compute_dprime_criterion_w_discrete_response(
        stimulus, response
    )

    assert np.allclose(array_dprime, 0)


def test_compute_PC_max_criterion_w_continuous_response_typical_cases():
    stimulus = {
        "df": pd.DataFrame({"amplitude": np.ones((4,)), "presence": np.ones((4,))})
    }
    stimulus["df"]["presence"][0:2] = 0

    response = {"df": pd.DataFrame({"presence": np.zeros((4,))})}
    response["df"]["presence"][1:3] = 1

    array_PC_max, array_criterion = compute_PC_max_criterion_w_continuous_response(
        stimulus, response
    )

    assert np.allclose(array_PC_max, 0.5)


def test_compute_PC_max_w_dprime_typical_cases():
    array_dprime = np.zeros((4,))
    PC_max = compute_PC_max_w_dprime(array_dprime)

    assert np.allclose(PC_max, 0.5)


def test_compute_dprime_w_PC_max_typical_cases():
    array_PC_max = np.ones((4,)) * 0.5
    array_dprime = compute_dprime_w_PC_max(array_PC_max)

    assert np.allclose(array_dprime, 0)


def test_negative_loglikelihood_w_parameter_edge_cases():
    x = [1, 0, 0]
    stimulus = {
        "df": pd.DataFrame({"amplitude": np.ones((3,)), "presence": np.ones((3,))})
    }
    response = {"df": pd.DataFrame({"presence": np.ones((3,))})}
    output = negative_loglikelihood_w_parameter(
        x, stimulus, response, unit_likelihood_abc
    )
    assert np.allclose(output, -3 * np.log(norm.cdf(0.5)))


def test_unit_likelihood_ab_edge_cases():
    x = [1, 0]
    stimulus = {"df": pd.DataFrame({"amplitude": [1], "presence": [1]})}
    response = {"df": pd.DataFrame({"presence": [1]})}
    assert unit_likelihood_ab(
        x, stimulus["df"].loc[0], response["df"].loc[0]
    ) == norm.cdf(0.5)

    stimulus["df"]["presence"] = 0
    assert unit_likelihood_ab(
        x, stimulus["df"].loc[0], response["df"].loc[0]
    ) == norm.cdf(-0.5)


def test_unit_likelihood_abc_edge_cases():
    x = [1, 0, 0]
    stimulus = {"df": pd.DataFrame({"amplitude": [1], "presence": [1]})}
    response = {"df": pd.DataFrame({"presence": [1]})}

    assert unit_likelihood_abc(
        x, stimulus["df"].loc[0], response["df"].loc[0]
    ) == norm.cdf(0.5)

    stimulus["df"]["presence"] = 0
    assert unit_likelihood_abc(
        x, stimulus["df"].loc[0], response["df"].loc[0]
    ) == norm.cdf(-0.5)


def test_unit_likelihood_gaussian_edge_cases():
    stimulus = {"df": pd.DataFrame({"amplitude": [0], "location": [1]})}
    response = {"df": pd.DataFrame({"location": [1]})}

    x = [1, 0]

    assert np.allclose(
        unit_likelihood_gaussian(x, stimulus["df"].loc[0], response["df"].loc[0]), 0.5
    )

    response = {"df": pd.DataFrame({"location": [2]})}
    assert np.allclose(
        unit_likelihood_gaussian(x, stimulus["df"].loc[0], response["df"].loc[0]), 0.5
    )

    stimulus = {"df": pd.DataFrame({"amplitude": [0], "location": [2]})}
    assert np.allclose(
        unit_likelihood_gaussian(x, stimulus["df"].loc[0], response["df"].loc[0]), 0.5
    )

    stimulus1 = {"df": pd.DataFrame({"amplitude": [1], "location": [1]})}
    response1 = {"df": pd.DataFrame({"location": [1]})}

    x1 = [5, 0.5]

    stimulus2 = {"df": pd.DataFrame({"amplitude": [1], "location": [2]})}
    response2 = {"df": pd.DataFrame({"location": [2]})}

    x2 = [5, -0.5]

    assert np.allclose(
        unit_likelihood_gaussian(x1, stimulus1["df"].loc[0], response1["df"].loc[0]),
        unit_likelihood_gaussian(x2, stimulus2["df"].loc[0], response2["df"].loc[0]),
    )


def test_unit_likelihood_uniform_edge_cases():
    stimulus = {"df": pd.DataFrame({"amplitude": [1], "location": [1]})}
    response = {"df": pd.DataFrame({"location": [1]})}

    x = [1, 0]
    assert np.allclose(
        unit_likelihood_uniform(x, stimulus["df"].loc[0], response["df"].loc[0]), 1
    )

    x = [1e12, 0]
    assert np.allclose(
        unit_likelihood_uniform(x, stimulus["df"].loc[0], response["df"].loc[0]), 0.5
    )

    response = {"df": pd.DataFrame({"location": [2]})}

    x = [1, 0]
    assert np.allclose(
        unit_likelihood_uniform(x, stimulus["df"].loc[0], response["df"].loc[0]), 0
    )

    stimulus1 = {"df": pd.DataFrame({"amplitude": [1], "location": [1]})}
    response1 = {"df": pd.DataFrame({"location": [1]})}

    x1 = [5, 0.5]

    stimulus2 = {"df": pd.DataFrame({"amplitude": [1], "location": [2]})}
    response2 = {"df": pd.DataFrame({"location": [2]})}

    x2 = [5, -0.5]

    assert np.allclose(
        unit_likelihood_uniform(x1, stimulus1["df"].loc[0], response1["df"].loc[0]),
        unit_likelihood_uniform(x2, stimulus2["df"].loc[0], response2["df"].loc[0]),
    )
