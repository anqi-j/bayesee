import numpy as np
from scipy.ndimage import label
from bayesee.perception import *


def modulate_std_keep_mean(x, scale_map):
    output = np.zeros_like(x)

    for scale_value in np.unique(scale_map):
        spot_map, n_group = label(scale_map == scale_value)

        for spot_label in np.unique(spot_map):
            if spot_label != 0:
                spot_mean = x[spot_map == spot_label].mean()
                spot_std = x[spot_map == spot_label].std()
                output[spot_map == spot_label] = normalize_mean_std(
                    x[spot_map == spot_label], spot_mean, spot_std * scale_value
                )

    return output
