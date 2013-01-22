from __future__ import division

import numpy as np
from numpy.ctypeslib import ndpointer

from ctypes import (sizeof, addressof, cast, POINTER, Structure,
                    c_int, c_uint8, c_float, c_double)
c_float_p = POINTER(c_float)

from vl_ctypes import LIB, c_to_np_types
from .utils import as_float_image


class VLDsiftKeypoint(Structure):
    _fields_ = [
        ('x', c_double),
        ('y', c_double),
        ('s', c_double),
        ('norm', c_double),
    ]


class VLDsiftDescriptorGeometry(Structure):
    _fields_ = [
        ('numBinT', c_int),
        ('numBinX', c_int),
        ('numBinY', c_int),

        ('binSizeX', c_int),
        ('binSizeY', c_int),
    ]


class VLDsiftFilter(Structure):
    _fields_ = [
        ('imWidth', c_int),
        ('imHeight', c_int),

        ('stepX', c_int),
        ('stepY', c_int),

        ('boundMinX', c_int),
        ('boundMinY', c_int),
        ('boundMaxX', c_int),
        ('boundMaxY', c_int),

        ('geom', VLDsiftDescriptorGeometry),

        ('useFlatWindow', c_int),
        ('windowSize', c_double),

        ('numFrames', c_int),
        ('descrSize', c_int),
        ('frames', POINTER(VLDsiftKeypoint)),
        ('descrs', c_float_p),

        ('numBinAlloc', c_int),
        ('numFrameAlloc', c_int),
        ('numGradAlloc', c_int),

        ('grads', POINTER(c_float_p)),
        ('convTmp1', c_float_p),
        ('convTmp2', c_float_p),
    ]
VLDsiftFilter_p = POINTER(VLDsiftFilter)

### functions in the shared object
# most of the utility functions are actually inlined and so not in the so...
vl_dsift_new = LIB['vl_dsift_new']
vl_dsift_new.restype = VLDsiftFilter_p
vl_dsift_new.argtypes = [c_int, c_int]

vl_dsift_new_basic = LIB['vl_dsift_new_basic']
vl_dsift_new_basic.restype = VLDsiftFilter_p
vl_dsift_new_basic.argtypes = [c_int, c_int, c_int, c_int]

vl_dsift_delete = LIB['vl_dsift_delete']
vl_dsift_delete.restype = None
vl_dsift_delete.argtypes = [VLDsiftFilter_p]

vl_dsift_process = LIB['vl_dsift_process']
vl_dsift_process.restype = None
vl_dsift_process.argtypes = [
    VLDsiftFilter_p,
    ndpointer(dtype=c_to_np_types[c_float]),
]

_vl_dsift_update_buffers = LIB['_vl_dsift_update_buffers']
_vl_dsift_update_buffers.restype = None
_vl_dsift_update_buffers.argtypes = [VLDsiftFilter_p]


# near-direct port of the c function
# TODO: vectorize...
def vl_dsift_transpose_descriptor(src, num_bin_t, num_bin_x, num_bin_y):
    dest = np.empty(num_bin_t * num_bin_x * num_bin_y, dtype=np.float32)
    for y in range(num_bin_y):
        for x in range(num_bin_x):
            offset = num_bin_t * (x + y * num_bin_x)
            offsetT = num_bin_t * (y + x * num_bin_x)

            for t in range(num_bin_t):
                tT = num_bin_t // 4 - t
                dest[offsetT + (tT + num_bin_t) % num_bin_t] = src[offset + t]
    return dest


def vl_dsift(data, fast=False, norm=False, bounds=None, size=3, step=1,
              window_size=None, float_descriptors=False):
    # make the image column-major, to be like matlab
    data = as_float_image(data, dtype=np.float32, order='F')
    if data.ndim != 2:
        raise TypeError("data should be a 2d array")

    if bounds is not None:
        if not len(bounds) == 4:
            raise TypeError("bounds should be None or a vector of 4 integers")

    if window_size is not None:
        assert np.isscalar(window_size) and window_size >= 0

    # construct the dsift object
    M, N = data.shape
    dsift_p = vl_dsift_new_basic(M, N, step, size)
    dsift = dsift_p.contents

    try:
        # set parameters
        if bounds:  # note that we're assuming the image is transposed
            dsift.boundMinX = max(bounds[1], 0)
            dsift.boundMinY = max(bounds[0], 0)
            dsift.boundMaxX = min(bounds[3], M - 1)
            dsift.boundMaxY = min(bounds[2], N - 1)
            _vl_dsift_update_buffers(dsift_p)

        dsift.useFlatWindow = fast

        if window_size is not None:
            dsift.windowSize = window_size

        # get calculated parameters
        num_frames = dsift.numFrames
        descr_size = dsift.descrSize
        geom = dsift.geom

        # do the actual processing
        vl_dsift_process(dsift_p, data)

        # copy computed results into output variables
        frames = dsift.frames
        descrs = dsift.descrs

        descr_type = c_float if float_descriptors else c_uint8
        out_descrs = np.zeros((descr_size, num_frames),
                              dtype=c_to_np_types[descr_type])
        out_frames = np.zeros((3 if norm else 2, num_frames),
                              dtype=c_to_np_types[c_double])

        # gross pointer arithmetic to get the relevant descriptor
        descrs_addr = addressof(descrs.contents)
        descrs_step = descr_size * sizeof(c_float)

        for k in range(num_frames):
            out_frames[:2, k] = [frames[k].y + 1, frames[k].x + 1]
            if norm:  # there's an implied / 2 in norm, because of clipping
                out_frames[2, k] = frames[k].norm

            # gross pointer arithmetic to get the relevant descriptor
            the_descr = cast(descrs_addr + k * descrs_step, c_float_p)
            transposed = vl_dsift_transpose_descriptor(
                the_descr,
                geom.numBinT, geom.numBinX, geom.numBinY)
            out_descrs[:, k] = np.minimum(512. * transposed, 255.)

        return out_frames, out_descrs

    finally:
        vl_dsift_delete(dsift_p)
