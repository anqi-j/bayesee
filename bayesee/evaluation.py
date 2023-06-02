import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def compute_PC_max_w_dprime(array_dprime):
    return norm.cdf(array_dprime / 2)


def compute_dprime_w_PC_max(array_PC_max):
    return 2 * norm.ppf(array_PC_max)


def compute_dprime_criterion_w_discrete_response(stimulus, response):
    amplitude = stimulus["df"]["amplitude"]
    presence = stimulus["df"]["presence"]
    response_presence = response["df"]["presence"]

    array_unique_amplitude = np.unique(amplitude)
    array_dprime = np.zeros_like(amplitude)
    array_criterion = np.zeros_like(amplitude)

    for unique_amplitude in array_unique_amplitude:
        index_unique = amplitude == unique_amplitude
        nBb = sum((presence == 1) & (response_presence == 1) & index_unique)
        nb = sum((presence == 1) & (index_unique))
        nBa = sum((presence == 0) & (response_presence == 1) & index_unique)
        na = sum((presence == 0) & (index_unique))

        if nBb == nb:
            pBb = 1 - 1 / (nb * 2)  # assume if double # trials, make 1 error
        elif nBb == 0:
            pBb = 1 / (nb * 2)  # assume if double # trials, make 1 correction
        else:
            pBb = nBb / nb

        if nBa == na:
            pBa = 1 - 1 / (na * 2)  # assume if double # trials, make 1 correction
        elif nBa == 0:
            pBa = 1 / (na * 2)  # assume if double # trials, make 1 error
        else:
            pBa = nBa / na

        array_dprime[index_unique] = norm.ppf(pBb) - norm.ppf(pBa)
        array_criterion[index_unique] = -0.5 * (norm.ppf(pBb) + norm.ppf(pBa))

    return array_dprime, array_criterion


def compute_PC_max_criterion_w_continuous_response(stimulus, response):
    amplitude = stimulus["df"]["amplitude"]
    presence = stimulus["df"]["presence"]
    decision_variable = response["df"]["presence"]

    array_unique_amplitude = np.unique(amplitude)
    array_PC_max = np.zeros_like(amplitude)
    array_criterion = np.zeros_like(amplitude)

    for unique_amplitude in array_unique_amplitude:
        index_unique = amplitude == unique_amplitude
        dva = decision_variable[(presence == 0) & index_unique]
        dvb = decision_variable[(presence == 1) & index_unique]
        na, nb = len(dva), len(dvb)
        ma, mb = dva.mean(), dvb.mean()
        va, vb = na * dva.var() / (na - 1), nb * dvb.var() / (nb - 1)

        print(ma, mb, va, vb)

        if np.allclose(va, vb) or np.allclose(ma, mb):
            criterion = (vb * ma + va * mb) / (va + vb)
            PC_max = 0.5
        elif mb > ma:
            criterion = (
                ma * vb
                - mb * va
                + np.sqrt(va * vb * ((mb - ma) ** 2 + (vb - va) * np.log(vb / va)))
            ) / (vb - va)
            PC_max = (
                norm.cdf((mb - criterion) / np.sqrt(vb))
                + norm.cdf((criterion - ma) / np.sqrt(va))
            ) / 2
        else:
            criterion = (
                ma * vb
                - mb * va
                - np.sqrt(va * vb * ((mb - ma) ** 2 + (vb - va) * np.log(vb / va)))
            ) / (vb - va)
            PC_max = (
                norm.cdf((ma - criterion) / np.sqrt(va))
                + norm.cdf((criterion - mb) / np.sqrt(vb))
            ) / 2

        array_PC_max[index_unique] = PC_max
        array_criterion[index_unique] = criterion

    return array_PC_max, array_criterion


def negative_loglikelihood_w_parameter(x, stimulus, response, unit_likelihood):
    negative_loglikelihood = 0

    metadata = {
        key: value for key, value in {**stimulus, **response}.items() if key != "df"
    }

    for index_trial in stimulus["df"].index:
        negative_loglikelihood -= np.log(
            unit_likelihood(
                x,
                stimulus["df"].loc[index_trial],
                response["df"].loc[index_trial],
                **metadata,
            )
        )

    return negative_loglikelihood


def unit_likelihood_ab(x, stimulus_df, response_df, **kwargs):
    amplitude = stimulus_df["amplitude"]
    presence = stimulus_df["presence"]
    response_presence = response_df["presence"]

    pBb = norm.cdf(0.5 * (amplitude / x[0]) ** x[1])
    pBa = norm.cdf(-0.5 * (amplitude / x[0]) ** x[1])

    if presence == 0:
        if response_presence == 0:
            return 1 - pBa
        else:
            return pBa
    else:
        if response_presence == 0:
            return 1 - pBb
        else:
            return pBb


def unit_likelihood_abc(x, stimulus_df, response_df, **kwargs):
    amplitude = stimulus_df["amplitude"]
    presence = stimulus_df["presence"]
    response_presence = response_df["presence"]

    pBb = norm.cdf(0.5 * (amplitude / x[0]) ** x[1] - x[2])
    pBa = norm.cdf(-0.5 * (amplitude / x[0]) ** x[1] - x[2])

    if presence == 0:
        if response_presence == 0:
            return 1 - pBa
        else:
            return pBa
    else:
        if response_presence == 0:
            return 1 - pBb
        else:
            return pBb


def unit_likelihood_gaussian(x, stimulus_df, response_df, **kwargs):
    amplitude = stimulus_df["amplitude"]
    location = stimulus_df["location"]
    response_location = response_df["location"]

    if location == 1:
        if response_location == 1:
            return norm.cdf((x[1] + amplitude) / x[0])
        else:
            return norm.cdf((-x[1] - amplitude) / x[0])
    else:
        if response_location == 1:
            return norm.cdf((x[1] - amplitude) / x[0])
        else:
            return norm.cdf((amplitude - x[1]) / x[0])


def unit_likelihood_uniform(x, stimulus_df, response_df, **kwargs):
    amplitude = stimulus_df["amplitude"]
    location = stimulus_df["location"]
    response_location = response_df["location"]

    if x[0] < amplitude and abs(x[1]) <= amplitude - x[0]:
        if location == response_location:
            return 1
        else:
            return 0

    if location == 1:
        if response_location == 1:
            return (
                np.arccos((-x[1] - amplitude) / x[0])
                + (amplitude + x[1])
                / (x[0] ** 2)
                * np.sqrt(x[0] ** 2 - (amplitude + x[1]) ** 2)
            ) / np.pi
        elif response_location == 2:
            return (
                np.arccos((x[1] + amplitude) / x[0])
                - (amplitude + x[1])
                / (x[0] ** 2)
                * np.sqrt(x[0] ** 2 - (amplitude + x[1]) ** 2)
            ) / np.pi
    elif location == 2:
        if response_location == 1:
            return (
                np.arccos((amplitude - x[1]) / x[0])
                - (amplitude - x[1])
                / (x[0] ** 2)
                * np.sqrt(x[0] ** 2 - (amplitude - x[1]) ** 2)
            ) / np.pi
        elif response_location == 2:
            return (
                np.arccos((x[1] - amplitude) / x[0])
                + (amplitude - x[1])
                / (x[0] ** 2)
                * np.sqrt(x[0] ** 2 - (amplitude - x[1]) ** 2)
            ) / np.pi
