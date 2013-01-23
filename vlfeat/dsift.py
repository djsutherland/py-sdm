from __future__ import division, print_function

import numpy as np
import numpy.ctypeslib as npc

from ctypes import cast, POINTER, Structure, c_int, c_float, c_double
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)

from vl_ctypes import LIB, c_to_np_types
from .utils import as_float_image

np_float = c_to_np_types[c_float]
np_double = c_to_np_types[c_double]


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
vl_dsift_process.argtypes = [VLDsiftFilter_p, npc.ndpointer(dtype=np_float)]

_vl_dsift_update_buffers = LIB['_vl_dsift_update_buffers']
_vl_dsift_update_buffers.restype = None
_vl_dsift_update_buffers.argtypes = [VLDsiftFilter_p]


# near-direct port of the c function
# should be vectorized, if we were going to call it more than once...
def vl_dsift_transpose_descriptor(dest, src, num_bin_t, num_bin_x, num_bin_y):
    for y in xrange(num_bin_y):
        for x in xrange(num_bin_x):
            offset = num_bin_t * (x + y * num_bin_x)
            offsetT = num_bin_t * (y + x * num_bin_x)
            for t in xrange(num_bin_t):
                tT = num_bin_t // 4 - t
                dest[offsetT + (tT + num_bin_t) % num_bin_t] = src[offset + t]


def vl_dsift(data, fast=False, norm=False, bounds=None, size=3, step=1,
              window_size=None, float_descriptors=False,
              verbose=False, matlab_style=True):
    '''
    Dense sift descriptors from an image.

    Returns:
        frames: num_frames x (2 or 3) matrix of x, y, (norm)
        descrs: num_frames x 128 matrix of descriptors
    '''
    if not matlab_style:
        import warnings
        warnings.warn("matlab_style=False gets different results than matlab, "
                      "not sure why or how incorrect they are.")

    order = 'F' if matlab_style else 'C'
    data = as_float_image(data, dtype=np.float32, order=order)
    if data.ndim != 2:
        raise TypeError("data should be a 2d array")

    if window_size is not None:
        assert np.isscalar(window_size) and window_size >= 0

    # construct the dsift object
    M, N = data.shape
    dsift_p = vl_dsift_new_basic(M, N, step, size)

    try:
        dsift = dsift_p.contents

        # set parameters
        if bounds is not None:
            if matlab_style:
                y0, x0, y1, x1 = bounds  # transposed
            else:
                x0, y0, x1, y1 = bounds
            dsift.boundMinX = max(x0, 0)
            dsift.boundMinY = max(y0, 0)
            dsift.boundMaxX = min(x1, M - 1)
            dsift.boundMaxY = min(y1, N - 1)
            _vl_dsift_update_buffers(dsift_p)

        dsift.useFlatWindow = fast

        if window_size is not None:
            dsift.windowSize = window_size

        # get calculated parameters
        descr_size = dsift.descrSize
        num_frames = dsift.numFrames
        geom = dsift.geom

        if verbose:
            pr = lambda *a, **k: print('vl_dsift:', *a, **k)
            pr("image size         [W, H] = [{}, {}]".format(N, M))
            x0 = dsift.boundMinX + 1
            y0 = dsift.boundMinY + 1
            x1 = dsift.boundMaxX + 1
            y1 = dsift.boundMaxY + 1
            bound_args = [y0, x0, y1, x1] if matlab_style else [x0, y0, x1, y1]
            pr("bounds:            [minX,minY,maxX,maxY] = [{}, {}, {}, {}]"
                .format(*bound_args))
            pr("subsampling steps: stepX={}, stepY={}".format(
                    dsift.stepX, dsift.stepY))
            pr("num bins:          [numBinT, numBinX, numBinY] = [{}, {}, {}]"
                .format(geom.numBinT, geom.numBinX, geom.numBinY))
            pr("descriptor size:   {}".format(descr_size))
            pr("bin sizes:         [binSizeX, binSizeY] = [{}, {}]".format(
                    geom.binSizeX, geom.binSizeY))
            pr("flat window:       {}".format(bool(fast)))
            pr("window size:       {}".format(dsift.windowSize))
            pr("num of features:   {}".format(num_frames))

        # do the actual processing
        vl_dsift_process(dsift_p, data)

        # copy frames' locations, norms out
        # the frames are a structure of just 4 doubles (VLDsiftKeypoint),
        # which luckily looks exactly like an array of doubles. :)
        # NOTE: this might be platform/compiler-dependent...but it works with
        #       the provided binaries on os x, at least
        frames_p = cast(dsift.frames, c_double_p)
        frames_p_a = npc.as_array(frames_p, shape=(num_frames, 4))
        cols = [1, 0] if matlab_style else [0, 1]
        if norm:
            cols.append(3)
        frames = np.require(frames_p_a[:, cols], requirements=['C', 'O'])

        # copy descriptors into a new array
        descrs_p = npc.as_array(dsift.descrs, shape=(num_frames, descr_size))
        descrs = descrs_p * 512
        assert descrs.flags.owndata
        np.minimum(descrs, 255, out=descrs)
        if not float_descriptors:
            descrs = descrs.astype(np.uint8)  # TODO: smarter about copying?
        if matlab_style:
            new_order = np.empty(descr_size, dtype=int)
            vl_dsift_transpose_descriptor(new_order, np.arange(descr_size),
                geom.numBinT, geom.numBinX, geom.numBinY)
            descrs = descrs[:, new_order]
        # the old, super-slow way:
        ## # gross pointer arithmetic to get the relevant descriptor
        ## descrs_addr = addressof(descrs.contents)
        ## descrs_step = descr_size * sizeof(c_float)
        ##
        ## for k in range(num_frames):
        ##     out_frames[:2, k] = [frames[k].y + 1, frames[k].x + 1]
        ##     if norm:  # there's an implied / 2 in norm, because of clipping
        ##         out_frames[2, k] = frames[k].norm
        ##
        ##     # gross pointer arithmetic to get the relevant descriptor
        ##     the_descr = cast(descrs_addr + k * descrs_step, c_float_p)
        ##     transposed = vl_dsift_transpose_descriptor(
        ##         the_descr,
        ##         geom.numBinT, geom.numBinX, geom.numBinY)
        ##     out_descrs[:, k] = np.minimum(512. * transposed, 255.)

        return frames, descrs

    finally:
        vl_dsift_delete(dsift_p)
