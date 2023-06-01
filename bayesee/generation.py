import numpy as np
from bayesee.interaction import *
from numpy.fft import fftn, fftshift, ifftshift, ifftn

# %% Deterministic


def euclidean_distance_exponential(size, center, exponent):
    output = np.zeros(size)

    for pixel_location in np.ndindex(size):
        delta_cartesian = np.array(pixel_location) - center
        euclidean_distance = np.linalg.norm(delta_cartesian)

        if exponent < 0 and euclidean_distance == 0:
            output[pixel_location] = 0
        else:
            output[pixel_location] = euclidean_distance**exponent

    return output


def sine_wave(size, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    fft_sine_wave = np.zeros(size, dtype=complex)

    center = np.array(size) // 2

    fft_sine_wave[*(center + frequencies)] = -1j
    fft_sine_wave[*(center - frequencies)] = 1j

    sine_wave = np.real(ifftn(ifftshift(fft_sine_wave)))

    return normalize_mean_std(sine_wave, mean, std)


def cosine_wave(size, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    fft_cosine_wave = np.zeros(size, dtype=complex)

    center = np.array(size) // 2

    fft_cosine_wave[*(center + frequencies)] = 1
    fft_cosine_wave[*(center - frequencies)] = 1

    cosine_wave = np.real(ifftn(ifftshift(fft_cosine_wave)))

    return normalize_mean_std(cosine_wave, mean, std)


def gaussian_2d(size, stds):
    row, col = size
    rho = 0

    if isinstance(stds, (int, float)):
        std1 = std2 = stds
    elif len(stds) == 2:
        std1, std2 = stds
    else:
        std1, std2, rho = stds

    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing="ij")
    z2 = (
        ((ii - row // 2) / std1) ** 2
        + ((jj - col // 2) / std2) ** 2
        - 2 * rho * (ii - row // 2) * (jj - col // 2) / (std1 * std2)
    )
    return np.exp(-z2 / (2 * (1 - rho**2)))


def square(size, center, widths):
    output = np.zeros(size)

    if isinstance(widths, (int, float)):
        widths = np.ones_like(center) * widths

    ranges = []
    for c, w in zip(center, widths):
        start = int(c - w / 2 + 1)
        end = int(c + w / 2 + 1)
        print(start, end)
        if start < 0 or end > c + 2:
            raise ValueError(
                f"Start={start} and end={end} must be within range:0-{c+2}."
            )

        ranges.append(slice(start, end, 1))

    output[tuple(ranges)] = 1

    return output


def circle(size, center, radii):
    output = np.zeros(size)

    if isinstance(radii, (int, float)):
        radii = np.ones_like(center) * radii

    for pixel_location in np.ndindex(size):
        delta_cartesian = np.array(pixel_location) - center
        if ((delta_cartesian / radii) ** 2).sum() < 1:
            output[pixel_location] = 1

    return output


def hann(size, center, width):
    euclidean_distance = euclidean_distance_exponential(size, center, 1)
    hann_function = np.cos(np.pi * euclidean_distance / (2 * width)) ** 2
    return hann_function * circle(size, center, width / 2)


# %% Probabilistic
def radial_asd_exponential(size, exponent, mean=0, std=1):
    # radial asd: radial amplitude spectral density
    white = np.random.normal(size=size)
    center = (np.array(size) - 1) / 2
    fft_filter = euclidean_distance_exponential(size, center, exponent)
    psd_filtered = filter_fft(white, fft_filter)
    return normalize_mean_std(psd_filtered, mean, std)


def probability_sample_dots(x, peak=1):
    if (x < 0).any():
        raise ValueError("Input must be non-negative.")

    normalized_x = peak * x / x.max()
    output = np.zeros_like(x)
    for pixel_location in np.ndindex(x.shape):
        if np.random.rand() < normalized_x[pixel_location]:
            output[pixel_location] = 1

    return output
