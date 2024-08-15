import numpy as np
from bayesee.perception import *
from numpy.fft import fftn, fftshift, ifftshift, ifftn
from scipy.signal import sawtooth, square

np_square = square

# %% Deterministic


def euclidean_distance_exponential(shape, center, exponent):
    grids = []
    for index_s, s in enumerate(shape):
        grids.append(np.linspace(-center[index_s], s - 1 - center[index_s], s))

    coordinates = np.meshgrid(*grids, indexing="ij")

    output = sum(c**2 for c in coordinates)

    if exponent < 0:
        output[output == 0] = 0
        output[output != 0] = output[output != 0] ** (exponent / 2)
    else:
        output = output ** (exponent / 2)

    return output


def orientation_angle_2d(shape, center):
    grids = [
        np.linspace(-center[index_s], s - 1 - center[index_s], s)
        for index_s, s in enumerate(shape)
    ]

    grid_di, grid_dj = np.meshgrid(*grids, indexing="ij")

    orientation = np.arctan2(-grid_di, -grid_dj) + np.pi

    return orientation


def sine_wave(shape, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    meshgrids = np.meshgrid(*(range(s) for s in shape), indexing="ij")

    sine_wave = -np.sum(
        np.sin(
            [
                2 * np.pi / s * (grid - s / 2) * freq
                for s, grid, freq in zip(shape, meshgrids, frequencies)
            ]
        ),
        axis=0,
    )

    return normalize_mean_std(sine_wave, mean, std)


def cosine_wave(shape, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    meshgrids = np.meshgrid(*(range(s) for s in shape), indexing="ij")

    cosine_wave = np.sum(
        np.cos(
            [
                2 * np.pi / s * (grid - s / 2) * freq
                for s, grid, freq in zip(shape, meshgrids, frequencies)
            ]
        ),
        axis=0,
    )

    return normalize_mean_std(cosine_wave, mean, std)


def sine_sawtooth_wave(shape, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    meshgrids = np.meshgrid(*(range(s) for s in shape), indexing="ij")

    sine_sawtooth_wave = -np.sum(
        sawtooth(
            [
                2 * np.pi / s * (grid - s / 2) * freq + np.pi / 2
                for s, grid, freq in zip(shape, meshgrids, frequencies)
            ],
            0.5,
        ),
        axis=0,
    )

    return normalize_mean_std(sine_sawtooth_wave, mean, std)


def cosine_sawtooth_wave(shape, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    meshgrids = np.meshgrid(*(range(s) for s in shape), indexing="ij")

    cosine_sawtooth_wave = -np.sum(
        sawtooth(
            [
                2 * np.pi / s * (grid - s / 2) * freq
                for s, grid, freq in zip(shape, meshgrids, frequencies)
            ],
            0.5,
        ),
        axis=0,
    )

    return normalize_mean_std(cosine_sawtooth_wave, mean, std)


def sine_square_wave(shape, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    meshgrids = np.meshgrid(*(range(s) for s in shape), indexing="ij")

    sine_square_wave = -np.sum(
        np_square(
            [
                2 * np.pi / s * (grid - s / 2) * freq
                for s, grid, freq in zip(shape, meshgrids, frequencies)
            ]
        ),
        axis=0,
    )

    return normalize_mean_std(sine_square_wave, mean, std)


def cosine_square_wave(shape, frequencies, mean=0, std=1):
    # frequencies: (vertical, horizontal) cycles/image
    meshgrids = np.meshgrid(*(range(s) for s in shape), indexing="ij")

    cosine_square_wave = -np.sum(
        np_square(
            [
                2 * np.pi / s * (grid - s / 2) * freq - np.pi / 2
                for s, grid, freq in zip(shape, meshgrids, frequencies)
            ],
            0.5,
        ),
        axis=0,
    )

    return normalize_mean_std(cosine_square_wave, mean, std)


def gaussian_2d(shape, stds):
    row, col = shape
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


def square(shape, center, widths):
    output = np.zeros(shape)

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


def circle(shape, center, radii):
    if isinstance(radii, (int, float)):
        radii = np.ones_like(center) * radii

    grids = []
    for index_s, s in enumerate(shape):
        grids.append(np.linspace(-center[index_s], s - 1 - center[index_s], s))

    coordinates = np.meshgrid(*grids, indexing="ij")

    distance = sum((c / radii[index_c]) ** 2 for index_c, c in enumerate(coordinates))

    output = np.zeros_like(distance)
    output[distance < 1] = 1

    return output


def hann(shape, center, width):
    # also called the raised cosine
    euclidean_distance = euclidean_distance_exponential(shape, center, 1)
    hann_function = np.cos(np.pi * euclidean_distance / width) ** 2
    return hann_function * circle(shape, center, width / 2)


def flat_top_hann(shape, center, head_width, foot_width):
    euclidean_distance = euclidean_distance_exponential(shape, center, 1)
    flat_top_distance = np.zeros(shape)
    flat_top_distance[euclidean_distance > head_width / 2] = (
        euclidean_distance - head_width / 2
    )[euclidean_distance > head_width / 2]
    flat_top_hann_function = (
        np.cos(np.pi * flat_top_distance / (foot_width - head_width)) ** 2
    )
    return flat_top_hann_function * circle(shape, center, foot_width / 2)


def radial_grid(shape, origin, unit_distance, angle_step):
    center = [origin]

    for angle in np.arange(0, 2 * np.pi, angle_step):
        focus = origin

        while (focus > 0).all() and focus[0] < shape[0] and focus[1] < shape[1]:
            focus = np.round(
                np.array(
                    [
                        focus[0] + unit_distance * np.cos(angle),
                        focus[1] + unit_distance * np.sin(angle),
                    ]
                )
            )
            center.append(focus)

    return center


def typical_grid(center, shape, origin, unit_distance, angle_step):
    if angle_step not in [np.pi / 3, np.pi / 2, 2 * np.pi / 3]:
        raise ValueError("Angle step must be pi/3, np.pi/2 or 2*pi/3.")

    if any(np.array_equal(origin, c) for c in center):
        return center

    center.append(origin)

    for angle in np.arange(0, 2 * np.pi, angle_step):
        candidate = np.round(
            np.array(
                [
                    origin[0] + unit_distance * np.cos(angle),
                    origin[1] + unit_distance * np.sin(angle),
                ]
            )
        )
        if (
            (candidate > 0).all()
            and candidate[0] < shape[0]
            and candidate[1] < shape[1]
        ):
            center = typical_grid(center, shape, candidate, unit_distance, angle_step)

    return center


def kernel_on_grid(shape, list_center, kernel):
    output = np.zeros(shape)
    kernel_shape = kernel.shape
    for center in list_center:
        slices = tuple(
            slice(c - k // 2, c + (k + 1) // 2)
            for c, k in zip(center.astype(int), kernel_shape)
        )
        output[slices] = kernel

    return output


# %% Probabilistic
def radial_asd_exponential(shape, exponent, mean=0, std=1):
    # radial asd: radial amplitude spectral density
    white = np.random.normal(size=shape)
    center = (np.array(shape) - 1) / 2
    fft_filter = euclidean_distance_exponential(shape, center, exponent)
    psd_filtered = filter_fft(white, fft_filter)
    return normalize_mean_std(psd_filtered, mean, std)


def probability_sample_dots(x, peak=1):
    # x: unnormalized sampling distribution
    if (x < 0).any():
        raise ValueError("Input must be non-negative.")

    normalized_x = peak * x / x.max()
    output = np.zeros_like(x)
    for pixel_location in np.ndindex(x.shape):
        if np.random.rand() < normalized_x[pixel_location]:
            output[pixel_location] = 1

    return output
