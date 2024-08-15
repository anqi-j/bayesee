import numpy as np
from scipy.ndimage import label
from bayesee.perception import *
from bayesee.variation import *


def test_modulate_std_keep_mean_typical_cases():
    x = np.array(
        [
            [-1, 1],
            [1, -1],
        ],
        dtype=float,
    )

    scale_map = np.array(
        [
            [0.5, 2],
            [0.5, 2],
        ]
    )
    assert (
        modulate_std_keep_mean(x, scale_map) == np.array([[-0.5, 2.0], [0.5, -2.0]])
    ).all()
