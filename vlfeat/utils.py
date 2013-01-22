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


def rgb2gray(img):
    """Converts an RGB image to grayscale using matlab's algorithm."""
    T = np.linalg.inv(np.array([
        [1.0,  0.956,  0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106,  1.703],
    ]))
    r_c, g_c, b_c = T[0]
    r, g, b = np.rollaxis(as_float_image(img), axis=-1)
    return r_c * r + g_c * g + b_c * b


# copied from skimage.color.rgb2hsv
def rgb2hsv(arr):
    """Converts an RGB image to HSV using scikit-image's algorithm."""
    arr = np.asanyarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("the input array must have a shape == (.,.,3)")
    arr = np.as_float_image(arr)

    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.] = 0.

    # -- H channel
    # red is max
    idx = (arr[:, :, 0] == out_v)
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:, :, 1] == out_v)
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = (arr[:, :, 2] == out_v)
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    np.seterr(**old_settings)

    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0
    return out
