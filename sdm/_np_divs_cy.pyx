from __future__ import division

cimport cython
from cython cimport view
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
                       _alpha_div as py_alpha_div,
                       _jensen_shannon_core as py_js_core)

cdef float fnan = float("NaN")
cdef float finf = float("inf")

@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.wraparound(False)
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
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _jensen_shannon_core(const int[:] Ks, int dim,
                               int min_i, const float[:] digamma_vals,
                               int num_q,
                               const float[:, ::1] rhos,
                               const float[:, ::1] nus,
                               const int[:] Ks_order, float min_sq_dist,
                               float[:] alphas_tmp, float[:] results) nogil:
    # NOTE: rhos contains all the neighbors up to max_K
    # NOTE: nus here is the "dists_out" array, which is a squared distance
    #       that hasn't been thresholded by min_dist
    cdef int i
    cdef int num_p = rhos.shape[0]

    cdef double t = 2 * num_p - 1
    cdef double p_wt = 1 / t
    cdef double q_wt = num_p / (num_q * t)

    cdef int max_K = rhos.shape[1]
    cdef int num_Ks = Ks.shape[0]

    cdef double alpha, max_wt = -1
    for i in range(num_Ks):
        alphas_tmp[i] = alpha = Ks[i] / (num_p + num_q - 1.)
        if alpha > max_wt:
            max_wt = alpha

    for i in range(num_Ks):
        results[i] = 0

    # mergesort rhos and nus
    # keeping track of the incremental weights until we hit each alpha
    cdef double curr_quantile, log_curr_dist, log_last_dist
    cdef double next_rho_log_dist, next_nu_log_dist
    cdef int next_rho, next_nu, next_alpha

    for i in range(num_p):
        curr_quantile = 0.
        next_alpha = 0
        log_curr_dist = log_last_dist = fnan

        next_rho = 0
        next_rho_log_dist = log(rhos[i, next_rho])

        next_nu = 0
        next_nu_log_dist = log(fmax(min_sq_dist, nus[i, next_nu])) / 2.

        while next_alpha < num_Ks:
            log_last_dist = log_curr_dist
            if next_rho_log_dist < next_nu_log_dist:
                log_curr_dist = next_rho_log_dist
                curr_quantile += p_wt
                next_rho += 1
                if next_rho == max_K:
                    next_rho_log_dist = finf
                else:
                    next_rho_log_dist = log(rhos[i, next_rho])
            else:
                log_curr_dist = next_nu_log_dist
                curr_quantile += q_wt
                next_nu += 1
                if next_nu == max_K:
                    next_nu_log_dist = finf
                else:
                    next_nu_log_dist = \
                        log(fmax(min_sq_dist, nus[i, next_nu])) / 2.

            while (next_alpha < num_Ks and
                   curr_quantile > alphas_tmp[Ks_order[next_alpha]]):
                results[Ks_order[next_alpha]] += (
                    dim * log_last_dist
                    - digamma_vals[next_rho + next_nu - 1 - min_i]
                ) / num_p
                next_alpha += 1


################################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
def _estimate_cross_divs(features, indices, rhos,
                         np.ndarray mask, funcs,
                         int[:] Ks, int max_K, bint save_all_Ks,
                         specs, int n_meta_only,
                         bint progressbar, int cores, float min_dist):
    # TODO: update to handle passing all Ks or only some
    cdef int a, i, j, k
    cdef int num_p, num_q, i_start, i_end, j_start, j_end

    cdef float[:, ::1] all_rhos_stacked, rhos_stacked

    if save_all_Ks:
        all_rhos_stacked = np.ascontiguousarray(np.vstack(rhos),
                                                dtype=np.float32)
        rhos_stacked = np.ascontiguousarray(
            np.asarray(all_rhos_stacked)[:, np.asarray(Ks) - 1])
    else:
        rhos_stacked = np.ascontiguousarray(np.vstack(rhos), dtype=np.float32)

    cdef float[:, ::1] all_features = \
        np.asarray(features._features, dtype=np.float32)
    cdef long[:] boundaries = features._boundaries

    cdef int n_bags = len(features)
    cdef int num_Ks = Ks.size
    cdef int dim = features.dim
    cdef float min_sq_dist = min_dist * min_dist

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

    cdef bint do_js = False
    cdef int js_min_i
    cdef float[:] js_digamma_vals
    cdef int[:] js_Ks_order
    cdef int js_pos

    for func, info in iteritems(funcs):
        assert isinstance(func, partial)
        assert not func.keywords
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

        elif real_func is py_js_core:
            do_js = True
            assert save_all_Ks
            the_Ks, the_dim, js_min_i, the_digamma_vals = func.args
            assert np.all(the_Ks == Ks)
            assert the_dim == dim
            assert the_digamma_vals.ndim == 1
            js_digamma_vals = np.asarray(the_digamma_vals, dtype=np.float32)

            js_Ks_order = np.argsort(Ks).astype(np.int32)

            js_pos, = info.pos
            if js_pos < 0:
                js_pos += num_funcs

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
    outputs[:, :, :, :] = fnan

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
    cdef float[:, ::1] alphas_tmp = np.empty((cores, num_Ks), dtype=np.float32)
    cdef int tid
    cdef long job_i, n_jobs = mask_is.shape[0]

    cdef object pbar
    cdef long jobs_since_last_tick_val
    cdef long * jobs_since_last_tick = &jobs_since_last_tick_val
    cdef uint8_t[:] is_done
    if progressbar:
        is_done = np.zeros(n_jobs, dtype=np.uint8)
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
            for job_i in prange(n_jobs, num_threads=cores, schedule='dynamic'):
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

                    # no need to set js self-values to nan, they already are
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

                    if do_js:
                        _jensen_shannon_core(Ks, dim,
                                             js_min_i, js_digamma_vals,
                                             num_q,
                                             all_rhos_stacked[i_start:i_end],
                                             dists_out[tid, :num_p, :],
                                             js_Ks_order, min_sq_dist,
                                             alphas_tmp[tid],
                                             outputs[i, j, js_pos, :])

                if progressbar:
                    is_done[job_i] = 1

        if progressbar:
            pbar.finish()

        return np.asarray(outputs)
    finally:
        free(index_array)


@cython.boundscheck(False)
@cython.wraparound(False)
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
