from __future__ import division

cimport cython
from cython cimport view
from cython.parallel import prange
from libc.math cimport log

from functools import partial

import numpy as np
cimport numpy as np

from .utils import lazy_range, izip, iteritems
from .mp_utils import progress

from ._np_divs import (_linear as py_linear,
                       kl as py_kl,
                       _alpha_div as py_alpha_div)


FLOAT = np.float64
ctypedef double FLOAT_T

OUTPUT = np.float32
ctypedef float OUTPUT_T


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _linear(FLOAT_T[:] Bs, int dim, int num_q,
                  FLOAT_T[:, ::1] nus,
                  OUTPUT_T[:] results) nogil:
    #   B / m * mean(nu ^ -dim)
    cdef int i, j
    cdef int num_p = nus.shape[0]
    cdef int num_Ks = results.shape[0]
    cdef FLOAT_T mean
    cdef FLOAT_T mdim = -dim

    for j in range(num_Ks):
        mean = 0
        for i in range(num_p):
            mean += (nus[i, j] ** mdim) / num_p
        results[j] = Bs[j] / num_q * mean


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void kl(int dim, int num_q,
             FLOAT_T[:, ::1] rhos, FLOAT_T[:, ::1] nus,
             OUTPUT_T[:] results) nogil:
    # dim * mean(log(nus) - log(rhos), axis=0) + log(num_q / (num_p - 1))

    cdef int i, j
    cdef int num_p = rhos.shape[0]
    cdef int num_Ks = results.shape[0]
    cdef FLOAT_T mean

    cdef FLOAT_T const = log(num_q / (<FLOAT_T> (num_p - 1)))

    for j in range(num_Ks):
        mean = 0
        for i in range(num_p):
            mean += (log(nus[i, j]) - log(rhos[i, j])) / num_p
        results[j] = max(0, dim * mean + const)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _alpha_div(FLOAT_T[:] omas, FLOAT_T[:, ::1] Bs,
                     int dim, int num_q,
                     FLOAT_T[:, ::1] rhos, FLOAT_T[:, ::1] nus,
                     int[:] poses, OUTPUT_T[:, ::1] results) nogil:
    cdef int i, j, k
    cdef int num_alphas = omas.shape[0]
    cdef int num_p = rhos.shape[0]
    cdef int num_Ks = rhos.shape[1]
    cdef FLOAT_T ratio, factor

    for i in range(num_alphas):
        for j in range(num_Ks):
            results[poses[i], j] = 0

    # the actual main estimate:
    #   mean( rho^(- dim * est alpha) nu^(- dim * est beta) )
    #   = mean( (rho / nu) ^ (dim * (1 - alpha)) )
    for k in range(num_p):
        for j in range(num_Ks):
            ratio = rhos[k, j] / nus[k, j]
            for i in range(num_alphas):
                results[poses[i], j] += ratio ** (dim * omas[i]) / num_p

    for i in range(num_alphas):
        factor = ((<FLOAT_T>(num_p - 1)) / num_q) ** omas[i]
        for j in range(num_Ks):
            results[poses[i], j] *= factor * Bs[i, j]
            if results[poses[i], j] < 0.:
                results[poses[i], j] = 0.


@cython.boundscheck(False)
def _estimate_cross_divs(features, indices, rhos,
                         mask, funcs, Ks, specs, int n_meta_only,
                         bint progressbar, int cores, float min_dist):
    cdef int i, j, p, s, start, end, rho_start, rho_end, nu_start, nu_end, num_q
    cdef long[:] boundaries
    cdef FLOAT_T[:, ::1] neighbors, rho, nu

    cdef FLOAT_T[:, ::1] rhos_stacked = \
        np.ascontiguousarray(np.vstack(rhos), dtype=FLOAT)

    cdef int n_bags = len(features)
    cdef int num_Ks = Ks.size
    cdef int max_K = np.max(Ks)
    cdef int dim = features.dim

    cdef int num_funcs = len(specs) + n_meta_only
    cdef OUTPUT_T[:, :, :, ::1] outputs = np.empty(
        (n_bags, n_bags, num_funcs, len(Ks)), dtype=OUTPUT)
    outputs[:, :, :, :] = np.nan

    # TODO: should just call functions that need self up here with rhos
    #       instead of computing nus and then throwing them out below
    any_run_self = False
    for func, info in iteritems(funcs):
        self_val = getattr(func, 'self_value', None)
        if self_val is not None:
            for i in range(n_bags):
                for p in info.pos:
                    outputs[i, i, p, :] = self_val
        else:
            any_run_self = True

    # hard-code to only allow calling alpha, kl, linear here for speed
    cdef bint do_linear = False
    cdef FLOAT_T[:] linear_Bs
    cdef int linear_pos

    cdef bint do_kl = False
    cdef int kl_pos

    cdef bint do_alpha = False
    cdef int alpha_num_alphas
    cdef FLOAT_T[:] alpha_omas
    cdef FLOAT_T[:, ::1] alpha_Bs
    cdef int[:] alpha_pos

    for func, info in iteritems(funcs):
        assert isinstance(func, partial)
        assert func.keywords is None
        real_func = func.func

        if real_func is py_linear:
            do_linear = True
            Bs, the_dim = func.args

            assert Bs.shape == (Ks.size,)
            linear_Bs = np.asarray(Bs, dtype=FLOAT)

            assert the_dim == dim

            linear_pos, = info.pos
            if linear_pos < 0:
                linear_pos += num_funcs

        elif real_func is py_kl:
            do_kl = True
            the_Ks, the_dim = func.args

            assert np.all(the_Ks == Ks)

            assert the_dim == dim

            kl_pos, = info.pos
            if kl_pos < 0:
                kl_pos += num_funcs

        elif real_func is py_alpha_div:
            do_alpha = True
            omas, Bs, the_dim = func.args

            alpha_omas = np.asarray(omas.ravel(), dtype=FLOAT)
            alpha_num_alphas = alpha_omas.size

            assert Bs.shape == (alpha_num_alphas, Ks.size)
            alpha_Bs = np.asarray(Bs, dtype=FLOAT)

            assert the_dim == dim

            alpha_pos = np.asarray(info.pos, dtype=np.int32)
            for i in range(alpha_pos.shape[0]):
                if alpha_pos[i] < 0:
                    alpha_pos[i] += num_funcs

        else:
            msg = "cython code can't handle function {}"
            raise ValueError(msg.format(real_func))

    indices_loop = progress()(indices) if progressbar else indices
    for i, index in enumerate(indices_loop):
        # Loop over rows of the output array.
        #
        # We want to search from most(?) of the other bags to this one, as
        # determined by mask and to avoid repeating nus.
        #
        # But we don't want to waste memory copying almost all of the features.
        #
        # So instead we'll run a separate NN search for each contiguous
        # subarray of the features. If they're too small, of course, this hurts
        # the parallelizability.
        #
        # TODO: is there a better scheme than this? use a custom version of
        #       nanoflann or something?

        num_q = features._n_pts[i]

        # make a boolean array of whether we want to do the ith bag
        do_bag = mask[i]
        if not any_run_self:
            do_bag = do_bag.copy()
            do_bag[i] = False

        # loop over contiguous sections where do_bag is True
        change_pts = np.hstack([0, np.diff(do_bag).nonzero()[0] + 1, n_bags])
        s = 0 if do_bag[0] else 1

        # TODO: make this whole loop nogil, since most of it is anyway
        # (this might require writing a little cython interface to flann
        #  instead of using the ctypes one....)
        for start, end in izip(change_pts[s::2], change_pts[s+1::2]):
            boundaries = features._boundaries[start:end+1]
            feats = features._features[boundaries[0]:boundaries[-1]]

            # find the nearest neighbors in features[i] from each of these bags
            neighbors = np.ascontiguousarray(
                    np.maximum(min_dist,
                        np.sqrt(index.nn_index(feats, max_K)[1][:, Ks - 1])),
                    dtype=FLOAT)

            with nogil:
                for j in prange(start, end, num_threads=cores):
                    rho_start = boundaries[j - start]
                    rho_end = boundaries[j - start + 1]
                    nu_start = rho_start - boundaries[0]
                    nu_end = rho_end - boundaries[0]

                    # rho = rhos_stacked[start_idx:end_idx]
                    # nu = neighbors[start_idx:end_idx]

                    if i == j:
                        # kl, alpha have already been done above
                        # nu and rho are the same, except with K off by one;
                        # use rho for both
                        if do_linear:
                            _linear(linear_Bs, dim, num_q,
                                    rhos_stacked[rho_start:rho_end],
                                    outputs[j, i, linear_pos, :])
                    else:
                        if do_linear:
                            _linear(linear_Bs, dim, num_q,
                                    neighbors[nu_start:nu_end],
                                    outputs[j, i, linear_pos, :])

                        if do_kl:
                            kl(dim, num_q,
                               rhos_stacked[rho_start:rho_end],
                               neighbors[nu_start:nu_end],
                               outputs[j, i, kl_pos, :])

                        if do_alpha:
                            _alpha_div(alpha_omas, alpha_Bs, dim, num_q,
                                       rhos_stacked[rho_start:rho_end],
                                       neighbors[nu_start:nu_end],
                                       alpha_pos, outputs[j, i, :, :])
    return np.asarray(outputs)
