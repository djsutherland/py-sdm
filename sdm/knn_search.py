'''
Convenience wrapper around FLANN to do kNN searches.
'''
from .utils import is_integer

import numpy as np

from pyflann import FLANN


def default_min_dist(dim):
    return min(1e-2, 1e-100 ** (1.0 / dim))


def pick_flann_algorithm(dim):
    return 'linear' if dim > 5 else 'kdtree_single'


def knn_search(K, x, y=None, min_dist=None, index=None, algorithm=None,
               return_indices=False, **kwargs):
    '''
    Calculates Euclidean distances to the first K closest elements of y
    for each x, which are row-instance data matrices.

    Returns a matrix whose (i, j)th element is the distance from the ith point
    in x to the (j+1)th nearest neighbor in y.

    If return_indices, also returns a matrix whose (i, j)th element is the
    identity of the (j+1)th nearest neighbor in y to the ith point in x.

    By default, clamps minimum distance to min(1e-2, 1e-100 ** (1/dim));
    setting min_dist to a number changes this value. Use 0 for no clamping.

    If index is passed, uses a preconstructed FLANN index for the elements of y
    (a FLANN() instance where build_index() has been run). Otherwise, constructs
    an index here and then deletes it, using the passed algorithm. By default,
    uses a single k-d tree for data with dimension 5 or lower, and brute-force
    search in higher dimensions (which give exact results). Any other keyword
    arguments are also passed to the FLANN() object.
    '''
    N, dim = x.shape
    if y is not None:
        M, dim2 = y.shape
        if dim != dim2:
            raise TypeError("x and y must have same second dimension")

    if not is_integer(K) or K < 1:
        raise TypeError("K must be a positive integer")

    if index is None:
        if algorithm is None:
            algorithm = pick_flann_algorithm(dim)
        index = FLANN(algorithm=algorithm, **kwargs)
        index.build_index(y)

    idx, dist = index.nn_index(x, K)

    idx = idx.astype(np.uint16)
    dist = np.sqrt(dist.astype(np.float64))

    # protect against identical points
    if min_dist is None:
        min_dist = default_min_dist(dim)
    if min_dist > 0:
        np.maximum(min_dist, dist, out=dist)

    return (dist, idx) if return_indices else dist
