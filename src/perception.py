import numpy as np
from numpy.fft import fftn, fftshift, ifftshift, ifftn
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

def filter_fft(x, fft_filter):
    # Unit for coordinates of fft_filter: cycles per image
    return np.real(ifftn(ifftshift(fftshift(fftn(x)) * fft_filter)))


def cosine_similarity(x, y):
    return np.dot(x.flatten(), y.flatten()) / (np.linalg.norm(x) * np.linalg.norm(y))

def fft_cosine_similarity(x, y):
    x_fft = fftn(x)
    y_fft = fftn(y)
    return np.abs(
        np.dot(x_fft.flatten(), y_fft.flatten())
        / (np.linalg.norm(x_fft) * np.linalg.norm(y_fft))
    )


def abs_cosine_similarity(x, y):
    x_fft = np.abs(fftn(x))
    y_fft = np.abs(fftn(y))
    return np.dot(x_fft.flatten(), y_fft.flatten()) / (
        np.linalg.norm(x_fft) * np.linalg.norm(y_fft)
    )


def angle_cosine_similarity(x, y):
    x_fft = np.angle(fftn(x))
    y_fft = np.angle(fftn(y))
    return np.dot(x_fft.flatten(), y_fft.flatten()) / (
        np.linalg.norm(x_fft) * np.linalg.norm(y_fft)
    )

def rmse(x, y):
    return np.sqrt(((x - y)**2).mean())

def mae(x, y):
    return (np.abs(x - y)).mean()

def pearson(x, y):
    return pearsonr(x.flatten(), y.flatten())[0]

def ssim(x,y):
    return structural_similarity(x, y, data_range=max(x.max(), y.max()) - min(x.min(), y.min()))

def normalize_min_max(x, mini=0, maxi=1):
    return ((maxi - mini) * x + (mini * x.max() - maxi * x.min())) / (x.max() - x.min())


def normalize_mean_std(x, mean=0, std=1):
    output = x.copy()
    output -= x.mean()

    if x.std() == 0:
        print("normalize_mean_std warning: input std = 0")
        return output + mean

    return std * output / output.std() + mean


def normalize_energy(x, energy=1):
    return x * np.sqrt(energy / np.dot(x.flatten(), x.flatten()))


def gamma_compress(x, bit=8, gamma=2.2):
    return (2**bit - 1) * (x / (2**bit - 1)) ** (1 / gamma)


def gamma_expand(x, bit=8, gamma=2.2):
    return (2**bit - 1) * (x / (2**bit - 1)) ** gamma


def discrete_value_mapping(source, mapping, shape=None):
    n_target_dim = len(mapping)

    minima = np.array(
        [
            min(
                [
                    int(mapping[dim][*source_index])
                    for source_index in np.ndindex(source.shape)
                ]
            )
            for dim in range(n_target_dim)
        ]
    )

    if shape is None:
        maxima = np.array(
            [
                max(
                    [
                        int(mapping[dim][*source_index])
                        for source_index in np.ndindex(source.shape)
                    ]
                )
                for dim in range(n_target_dim)
            ]
        )

        shape = maxima - minima + 2

    target = np.zeros(shape)
    count = np.zeros_like(target, dtype=int)
    for source_index in np.ndindex(source.shape):
        target_index = [
            int(mapping[dim][*source_index] - minima[dim])
            for dim in range(n_target_dim)
        ]

        if count[*target_index] == 0:
            target[*target_index] = source[*source_index]
        # else:
        #     target[*target_index] = (
        #         target[*target_index] * count[*target_index] + source[*source_index]
        #     ) / (count[*target_index] + 1)

        count[*target_index] += 1

    return target, count, minima


def human_contrast_sensitivity_function(
    shape,
    ppd,
    pupil_diameter=4,
    wavelength=555,
    surround_strength=0.85,
    surround_size=0.15,
    center_size=0.065,
    surround_exponent=2,
):
    # Default values from: Watson, A. B., & Ahumada, A. J. (2005). A standard model for foveal detection of spatial contrast. Journal of vision, 5(9), 6-6. Watson, A. B. (2013). A formula for the mean human optical modulation transfer function as a function of pupil size. Journal of Vision, 13(6), 18-18.
    # Unit for surround_size, center_size, pupil diameter: mm
    # Unit for coordinates of csf: cycles per image

    row, col = shape
    xx, yy = np.meshgrid(range(row), range(col), sparse=True)
    xx_img = xx - row // 2
    yy_img = yy - col // 2
    xx_deg = xx_img / (row / ppd)
    yy_deg = yy_img / (col / ppd)
    spatial_frequency_cpd = np.hypot(xx_deg, yy_deg)

    u0 = pupil_diameter * np.pi * 10**6 / (wavelength * 180)
    u1 = 21.95 - 5.512 * pupil_diameter + 0.3922 * pupil_diameter**2
    uh = spatial_frequency_cpd / u0
    D = (np.arccos(uh) - uh * np.sqrt(1 - uh**2)) * 2 / np.pi

    optical_transfer_function = np.sqrt(D) * (
        1 + (spatial_frequency_cpd / u1) ** 2
    ) ** (-0.62)
    csf = (
        optical_transfer_function
        * (
            1
            - surround_strength
            * np.exp(-surround_size * spatial_frequency_cpd**surround_exponent)
        )
        * np.exp(-center_size * spatial_frequency_cpd)
    )

    csf[row // 2, col // 2] = 0  # physically meaningless

    return csf
