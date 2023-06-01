import numpy as np
from numpy.fft import fftn, fftshift, ifftshift, ifftn


def filter_fft(x, fft_filter):
    # Unit for coordinates of fft_filter: cycles per image
    return np.real(ifftn(ifftshift(fftshift(fftn(x)) * fft_filter)))


def normalize_mean_std(x, mean=0, std=1):
    output = x.copy()
    output -= output.mean()
    return std * output / output.std() + mean


def normalize_energy(x, energy=1):
    return x / np.sqrt(energy * np.dot(x.flatten(), x.flatten()))


def gamma_compress(x, bit=8, gamma=2.2):
    return (2**bit - 1) * (x / (2**bit - 1)) ** (1 / gamma)


def gamma_expand(x, bit=8, gamma=2.2):
    return (2**bit - 1) * (x / (2**bit - 1)) ** gamma


def human_contrast_sensitivity_function(
    size,
    ppd,
    pupil_diameter=4,
    wavelength=555,
    surround_strength=0.85,
    surround_size=0.15,
    center_size=0.065,
    surround_exponent=2,
):
    # Default values from: Watson, A. B., & Ahumada, A. J. (2005). A standard model for foveal detection of spatial contrast. Journal of vision, 5(9), 6-6.
    # Unit for surround_size, center_size, pupil diameter: mm
    # Unit for coordinates of csf: cycles per image

    row, col = size
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
