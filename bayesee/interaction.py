import numpy as np
from numpy.fft import fftn, fftshift, ifftshift, ifftn

def filter_fft(x, fft_filter):
    return np.real(ifftn(ifftshift(fftshift(fftn(x))*fft_filter)))