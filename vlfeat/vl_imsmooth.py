from __future__ import division

import numpy as np
from numpy.ctypeslib import ndpointer

from ctypes import c_double, c_float, c_int, c_uint
from vl_ctypes import (LIB, c_to_np_types, np_to_c_types, vl_size, vl_index,
                       vl_epsilon_f)
from .utils import as_float_image

_imconvcol = {}
for name, datatype in [('d', c_double), ('f', c_float)]:
    _imconvcol[datatype] = fn = LIB['vl_imconvcol_v' + name]

    dtype = c_to_np_types[datatype]
    fn.restype = None
    fn.argtypes = [
        ndpointer(dtype=dtype, flags='WRITEABLE'), vl_size,  # dst, dst_stride
        ndpointer(dtype=dtype),  # src
        vl_size, vl_size, vl_size,  # src_width, src_height, src_stride
        ndpointer(dtype=dtype), vl_index, vl_index,  # filt, _begin, _end
        c_int, c_uint  # step, flags
    ]


PADDINGS = {'continuity': 1, 'zero': 0}
TRANSPOSE = 4


def vl_imsmooth(image, sigma, step=1, padding='continuity'):
    # unlike matlab interface, only does gaussian kernel

    if not np.isscalar(sigma):
        raise TypeError("sigma must be a scalar float")

    try:
        padding = PADDINGS[padding]
    except KeyError:
        raise ValueError("unknown padding {!r}; expected one of {}".format(
            padding, ', '.join(PADDINGS.keys())))
    flags = padding | TRANSPOSE

    if not np.isscalar(step) or \
            not issubclass(np.asarray(step).dtype.type, np.integer) or \
            step < 1:
        raise TypeError("step should be a positive int, not {!r}".format(step))

    # make sure image is col-major, to be exactly the same as the mex function
    # TODO: don't bother with transposing...?
    image = as_float_image(image, order='F')
    ndim = image.ndim

    dtype = image.dtype
    c_type = np_to_c_types[dtype]
    imconvcol = _imconvcol[c_type]

    if c_type not in (c_float, c_double):
        raise TypeError("image must be either float or double")
    elif not 2 <= ndim <= 3:
        raise TypeError("image must be a 2d or 3d array")

    if sigma < 0.01 and step == 1:
        return np.copy(image)

    if ndim == 2:
        image = image.reshape(image.shape + (1,), order='F')

    in_rows, in_cols, channels = image.shape
    out_rows, out_cols = (np.asarray(image.shape[:2]) - 1) // step + 1

    output = np.zeros((out_rows, out_cols, channels), dtype=dtype, order='F')

    # obtained from the input image by convolving and downsampling along the
    # height, saving the result transposed
    temp = np.empty(in_rows * out_cols, dtype=dtype)

    W = int(np.ceil(4.0 * sigma))
    filter = np.arange(-W, W + 1).astype(dtype)
    filter = np.exp(-0.5 * (filter / (sigma + vl_epsilon_f)) ** 2)
    filter /= filter.sum()

    for k in range(channels):
        image_channel = image[:, :, k]
        output_channel = output[:, :, k]
        imconvcol(temp, out_cols,
                  image_channel, in_rows, in_cols, in_rows,
                  filter, -W, W, step, flags)
        imconvcol(output_channel, out_rows,
                  temp, out_cols, in_rows, out_cols,
                  filter, -W, W, step, flags)

    if ndim == 2:
        output = np.squeeze(output)
    return output
