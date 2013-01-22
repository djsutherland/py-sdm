from __future__ import division
import numpy as np


def as_float_image(image, dtype=None, order=None):
    if image.dtype.kind in ('u', 'i'):
        bytes = image.dtype.itemsize
        if dtype is None:
            dtype = np.float32 if bytes <= 3 else np.float64
        max = 2 ** (8 * (bytes - (1 if image.dtype.kind == 'i' else 0))) - 1
        return np.asarray(image, dtype=dtype, order=order) / max
    else:
        assert np.max(image) <= 1
        assert np.max(image) >= 0
        return np.asarray(image, dtype=dtype, order=order)
