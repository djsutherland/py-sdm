# distutils: language = c++
# distutils: sources = _flann_quantile.cpp

import numpy as np
cimport numpy as np
cimport cython

from cyflann.flann cimport flann_index_t, FLANNParameters
from cyflann.index cimport FLANNIndex, FLANNParameters as CyFLANNParameters
from ._flann_quantile cimport flann_quantile_search_float


def quantile_search(FLANNIndex index, double[:] weights, qpts,
                    weight_targets=[], n_neighbors=[],
                    bint le_weight=False, **kwargs):
    cdef int i
    if index._this is NULL:
        raise ValueError("need to build index first")

    cdef float[:, ::1] the_qpts = index._check_array(qpts)

    cdef int npts = index._data.shape[0], dim = index._data.shape[1]
    cdef int nqpts = the_qpts.shape[0], qdim = the_qpts.shape[1]
    if qdim != dim:
        raise TypeError("data is dim {}, query is dim {}".format(dim, qdim))

    if weights.ndim != 1:
        msg = "weights should be 1d, is {}d"
        raise TypeError(msg.format(weights.ndim))
    if weights.shape[0] != npts:
        msg = "weights should have {} entries, has {}"
        raise TypeError(msg.format(npts, weights.shape[0]))

    cdef int[:] the_n_neighbors = np.asarray(n_neighbors, dtype=np.int32)
    cdef int max_neighbor = -1
    if the_n_neighbors.ndim != 1:
        msg = "n_neighbors should be 1d, is {}d"
        raise TypeError(msg.format(the_n_neighbors.ndim))
    if the_n_neighbors.shape[0] > 0:
        for i in range(the_n_neighbors.shape[0]):
            if the_n_neighbors[i] <= 0:
                msg = "can't get the {}-th nearest neighbor"
                raise ValueError(msg.format(the_n_neighbors[i]))
            elif the_n_neighbors[i] > npts:
                msg = "asking for {} neighbors from a set of size {}"
                raise ValueError(msg.format(max_neighbor, npts))

    cdef double[:] the_weight_targets = \
            np.asarray(weight_targets, dtype=np.float64)
    if the_weight_targets.ndim != 1:
        msg = "weight_targets should be 1d, is {}d"
        raise TypeError(msg.format(the_weight_targets.ndim))

    cdef double max_weight, total_weight
    if the_weight_targets.shape[0] > 0:
        max_weight = np.max(the_weight_targets)
        total_weight = 0
        for i in range(weights.shape[0]):
            total_weight += weights[i]
            if weights[i] < 0:
                msg = "negative weight: {} = {}"
                raise ValueError(msg.format(i, weights[i]))
        if max_weight > total_weight:
            msg = "asking for {} weight from a set with total weight {}"
            raise ValueError(msg.format(max_weight, total_weight))

    index.params.update(**kwargs)

    cdef tuple shape = (nqpts,
                        the_weight_targets.shape[0] + the_n_neighbors.shape[0])
    cdef np.ndarray idx = np.empty(shape, dtype=np.int32)
    cdef np.ndarray dists = np.empty(shape, dtype=np.float32)

    cdef int res = _quantile_search(
        index=index, weights=weights, qpts=the_qpts,
        weight_targets=the_weight_targets, n_neighbors=the_n_neighbors,
        le_weight=le_weight,
        idx=idx, dists=dists)

    if shape[1] == 1:
        return idx[:, 0], dists[:, 0]
    else:
        return idx, dists


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _quantile_search(FLANNIndex index, double[:] weights,
                          float[:, ::1] qpts,
                          double[:] weight_targets, int[:] n_neighbors,
                          bint le_weight,
                          int[:, ::1] idx, float[:, ::1] dists) nogil:
    return flann_quantile_search_float(
        index._this, weights=&weights[0],
        testset=&qpts[0, 0], tcount=qpts.shape[0],
        indices=&idx[0, 0], dists=&dists[0, 0],
        weight_targets=&weight_targets[0], weightcount=weight_targets.shape[0],
        n_neighbors=&n_neighbors[0], neighborcount=n_neighbors.shape[0],
        le_weight=le_weight, flann_params=&index.params._this)
