from cyflann.index cimport FLANNIndex

cdef int _quantile_search(FLANNIndex index, double[:] weights,
                          float[:, ::1] qpts,
                          double[:] weight_targets, int[:] n_neighbors,
                          bint le_weight,
                          int[:, ::1] idx, float[:, ::1] dists) nogil
