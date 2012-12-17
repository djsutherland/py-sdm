'''
Helper functions to compute k-nearest-neighbor distances between sets.
Intended for use with the divergence estimators in div_estimation.py.
'''
from __future__ import division, absolute_import

import warnings

import numpy as np

try:
    from pyflann import FLANN
    searcher = FLANN()
except ImportError:
    warnings.warn('Cannot find FLANN. KNN searches will be much slower.')
    searcher = None

from .utils import col, is_integer


def l2_dist_sq(A, B):
    '''
    Calculates pairwise squared Euclidean distances between points in A and
    in B, which are row-instance data matrices.
    Returns a matrix whose (i,j)th element is the distance from the ith point
    in A to the jth point in B.
    '''
    return -2 * np.dot(A, B.T) + col((A**2).sum(1)) + (B**2).sum(1)


def knn_search(x, y, K, min_dist=None):
    '''
    Calculates distances to the first K closest elements of y for each x,
    which are row-instance data matrices.

    Returns a matrix whose (i,j)th element is the distance from the ith point
    in x to the jth-closest point in y.

    By default, clamps minimum distance to min(1e-2, 1e-100 ** (1/dim));
    setting min_dist to a number changes this value. Use 0 for no clamping.
    '''
    N, dim = x.shape
    M, dim2 = y.shape
    if dim != dim2:
        raise TypeError("x and y must have same second dimension")
    if not is_integer(K) and K >= 1:
        raise TypeError("K must be a positive integer")

    if searcher is not None:
        algorithm = 'linear' if dim > 5 else 'kdtree'
        idx, dist = searcher.nn(y, x, K, algorithm=algorithm)
    else:
        D = l2_dist_sq(x, y)
        idx = np.argsort(D, 1)[:, :K]
        dist = D[np.repeat(col(np.arange(N)), K, axis=1), idx]

    idx = idx.astype('uint16')
    dist = np.sqrt(dist.astype('float64'))

    # protect against identical points
    if min_dist is None:
        min_dist = min(1e-2, 1e-100**(1.0/dim))
    elif min_dist <= 0:
        return dist, idx
    return np.maximum(min_dist, dist), idx
