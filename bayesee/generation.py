import numpy as np
from bayesee.interaction import *

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

def radial_asd_exponential(size, exponent, mean=0, std=1):
    # radial asd: radial amplitude spectral density
    white = np.random.normal(size=size)
    center = (np.array(size)-1) / 2
    fft_filter = euclidean_distance_exponential(size, center, exponent)
    psd_filtered = filter_fft(white, fft_filter)
    psd_filtered -= psd_filtered.mean()
    output = std*psd_filtered/psd_filtered.std()+mean
    return output
