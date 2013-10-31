from __future__ import division

cimport cython
from cython cimport view, integral
from cython.parallel import prange, threadid
from libc.stdlib cimport malloc, free
from libc.math cimport log, sqrt, fmax
from cpython.exc cimport PyErr_CheckSignals

from functools import partial

import numpy as np
cimport numpy as np
from numpy cimport uint8_t

from cyflann.flann cimport flann_index_t, FLANNParameters, \
                           flann_find_nearest_neighbors_index_float
from cyflann.index cimport FLANNIndex, FLANNParameters as CyFLANNParameters

from .utils import lazy_range, izip, iteritems
from .mp_utils import progress

from ._np_divs import (_linear as py_linear,
                       kl as py_kl,
                       _alpha_div as py_alpha_div)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _linear(float[:] Bs, int dim, int num_q,
                  float[:, ::1] nus,
                  float[:] results) nogil:
    #   B / m * mean(nu ^ -dim)
    cdef int i, j
    cdef int num_p = nus.shape[0]
    cdef int num_Ks = results.shape[0]
    cdef float mean
    cdef float mdim = -dim

    for j in range(num_Ks):
        mean = 0
        for i in range(num_p):
            mean += (nus[i, j] ** mdim) / num_p
        results[j] = Bs[j] / num_q * mean


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void kl(int dim, int num_q,
             float[:, ::1] rhos, float[:, ::1] nus,
             float[:] results) nogil:
    # dim * mean(log(nus) - log(rhos), axis=0) + log(num_q / (num_p - 1))

    cdef int i, j
    cdef int num_p = rhos.shape[0]
    cdef int num_Ks = results.shape[0]
    cdef float mean

    cdef float const = log(num_q / (<float> (num_p - 1)))

    for j in range(num_Ks):
        mean = 0
        for i in range(num_p):
            mean += (log(nus[i, j]) - log(rhos[i, j])) / num_p
        results[j] = max(0, dim * mean + const)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _alpha_div(float[:] omas, float[:, ::1] Bs,
                     int dim, int num_q,
                     float[:, ::1] rhos, float[:, ::1] nus,
                     int[:] poses, float[:, ::1] results) nogil:
    cdef int i, j, k
    cdef int num_alphas = omas.shape[0]
    cdef int num_p = rhos.shape[0]
    cdef int num_Ks = rhos.shape[1]
    cdef float ratio, factor

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
        factor = ((<float>(num_p - 1)) / num_q) ** omas[i]
        for j in range(num_Ks):
            results[poses[i], j] *= factor * Bs[i, j]
            if results[poses[i], j] < 0.:
                results[poses[i], j] = 0.

# TODO: jensen_shannon_core

################################################################################


@cython.boundscheck(False)
def _estimate_cross_divs(features, indices, rhos,
                         np.ndarray mask, funcs, integral[:] Ks,
                         specs, int n_meta_only,
                         bint progressbar, int cores, float min_dist):
    # TODO: update to handle passing all Ks or only some
    cdef int a, i, j, k
    cdef int num_p, num_q, i_start, i_end, j_start, j_end

    cdef float[:, ::1] rhos_stacked = \
        np.ascontiguousarray(np.vstack(rhos), dtype=np.float32)
    cdef float[:, ::1] all_features = \
        np.asarray(features._features, dtype=np.float32)
    cdef long[:] boundaries = features._boundaries

    cdef int n_bags = len(features)
    cdef int num_Ks = Ks.size
    cdef int max_K = np.max(Ks)
    cdef int dim = features.dim

    ############################################################################
    ### Handle the funcs we have.
    # Hard-coded to only allow calling alpha, kl, linear here, for speed.

    cdef int num_funcs = len(specs) + n_meta_only

    cdef bint do_linear = False
    cdef float[:] linear_Bs
    cdef int linear_pos

    cdef bint do_kl = False
    cdef int kl_pos

    cdef bint do_alpha = False
    cdef int alpha_num_alphas
    cdef float[:] alpha_omas
    cdef float[:, ::1] alpha_Bs
    cdef int[:] alpha_pos

    for func, info in iteritems(funcs):
        assert isinstance(func, partial)
        assert func.keywords is None
        real_func = func.func

        if real_func is py_linear:
            do_linear = True
            Bs, the_dim = func.args

            assert Bs.shape == (Ks.size,)
            linear_Bs = np.asarray(Bs, dtype=np.float32)

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

            alpha_omas = np.asarray(omas.ravel(), dtype=np.float32)
            alpha_num_alphas = alpha_omas.size

            assert Bs.shape == (alpha_num_alphas, Ks.size)
            alpha_Bs = np.asarray(Bs, dtype=np.float32)

            assert the_dim == dim

            alpha_pos = np.asarray(info.pos, dtype=np.int32)
            for i in range(alpha_pos.shape[0]):
                if alpha_pos[i] < 0:
                    alpha_pos[i] += num_funcs

        # TODO: jensen_shannon_core

        else:
            msg = "cython code can't handle function {}"
            raise ValueError(msg.format(real_func))

    ############################################################################

    # use params with cores=1
    cdef FLANNParameters params = (<CyFLANNParameters> indices[0].params)._this
    params.cores = 1

    # figure out which matrix elements we need to do
    nonzero_i, nonzero_j = mask.nonzero()
    cdef int[:] mask_is = nonzero_i.astype(np.int32), \
                mask_js = nonzero_j.astype(np.int32)

    # the results variable
    cdef float[:, :, :, ::1] outputs = np.empty(
        (n_bags, n_bags, num_funcs, len(Ks)), dtype=np.float32)
    outputs[:, :, :, :] = np.nan

    # temporay working variables
    cdef int max_pts = np.max(features._n_pts)

    # work buffer for each thread; first axis is for a thread.
    # done since cython doesn't currently support thread-local memoryviews :|
    cdef int[:, :, ::1] idx_out = \
        np.empty((cores, max_pts, max_K), dtype=np.int32)
    cdef float[:, :, ::1] dists_out = \
        np.empty((cores, max_pts, max_K), dtype=np.float32)
    cdef float[:, :, ::1] neighbors = \
        np.empty((cores, max_pts, num_Ks), dtype=np.float32)
    cdef int tid
    cdef long job_i, n_jobs = mask_is.shape[0]

    cdef object pbar
    cdef long jobs_since_last_tick_val
    cdef long * jobs_since_last_tick = &jobs_since_last_tick_val
    cdef uint8_t[:] is_done
    if progressbar:
        is_done = np.empty(n_jobs, dtype=np.uint8)
        pbar = progress(maxval=n_jobs)
        pbar.start()

    # make a C array of pointers to indices, so we can get it w/o the GIL
    cdef flann_index_t * index_array = <flann_index_t *> malloc(
                n_bags * sizeof(flann_index_t))
    if not index_array:
        raise MemoryError()
    try:
        # populate the index_array
        for i in range(n_bags):
            index_array[i] = (<FLANNIndex> indices[i])._this

        with nogil:
            for job_i in prange(n_jobs, num_threads=cores, schedule='guided'):
                tid = threadid()
                i = mask_is[job_i]
                j = mask_js[job_i]

                if tid == 0:
                    with gil:
                        PyErr_CheckSignals()  # allow ^C to interrupt us
                    if progressbar:
                        handle_pbar(pbar, jobs_since_last_tick, is_done)

                i_start = boundaries[i]
                i_end = boundaries[i + 1]
                num_p = i_end - i_start

                if i == j:
                    if do_linear:
                        _linear(linear_Bs, dim, num_p,
                                rhos_stacked[i_start:i_end],
                                outputs[i, j, linear_pos, :])
                    if do_kl:
                        outputs[i, j, kl_pos, :] = 0

                    if do_alpha:
                        for k in range(alpha_pos.shape[0]):
                            outputs[i, j, alpha_pos[k], :] = 1
                else:
                    j_start = boundaries[j]
                    j_end = boundaries[j + 1]
                    num_q = j_end - j_start

                    # do the nearest neighbor search from p to q
                    flann_find_nearest_neighbors_index_float(
                        index_id=index_array[j],
                        testset=&all_features[i_start, 0],
                        trows=num_p,
                        indices=&idx_out[tid, 0, 0],
                        dists=&dists_out[tid, 0, 0],
                        nn=max_K,
                        flann_params=&params)
                    for a in range(num_p):
                        for k in range(num_Ks):
                            neighbors[tid, a, k] = fmax(min_dist,
                                       sqrt(dists_out[tid, a, Ks[k] - 1]))

                    if do_linear:
                        _linear(linear_Bs, dim, num_q,
                                neighbors[tid, :num_p, :],
                                outputs[i, j, linear_pos, :])

                    if do_kl:
                        kl(dim, num_q,
                           rhos_stacked[i_start:i_end],
                           neighbors[tid, :num_p, :],
                           outputs[i, j, kl_pos, :])

                    if do_alpha:
                        _alpha_div(alpha_omas, alpha_Bs, dim, num_q,
                                   rhos_stacked[i_start:i_end],
                                   neighbors[tid, :num_p, :],
                                   alpha_pos, outputs[i, j, :, :])

                if progressbar:
                    is_done[job_i] = 1

        if progressbar:
            pbar.finish()

        return np.asarray(outputs)
    finally:
        free(index_array)


@cython.boundscheck(False)
cdef bint handle_pbar(object pbar, long * jobs_since_last_tick,
                      uint8_t[:] is_done) nogil except 1:
    jobs_since_last_tick[0] += 1

    cdef long done_count = 0

    # TODO: tweak this number?
    if jobs_since_last_tick[0] >= 20:
        for k in range(is_done.shape[0]):
            if is_done[k]:
                done_count += 1

        with gil:
            pbar.update(done_count)
        jobs_since_last_tick[0] = 0
