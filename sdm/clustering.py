import operator as op

import numpy as np
from scipy import sparse

from .utils import is_integer_type, lazy_range

# A basic kernel k-means implementation, ported from:
# mathworks.com/matlabcentral/fileexchange/26182-kernel-k-means/content/knkmeans.m
# TODO: needs checking (both correctness of port and correctness of algorithm)

# TODO: kmeans++ or other initialization schemes
def kn_kmeans(K, init):
    K = np.asarray(K)
    m, n = K.shape
    assert m == n

    label = None
    if is_integer_type(init):
        if np.isscalar(init):
            # scalar init means that number of clusters
            label = np.round(np.linspace(0, init-1, num=n))
            np.random.shuffle(label)
        elif np.squeeze(init).shape == (n,):
            label = np.squeeze(init)
    if label is None:
        raise ValueError("init should be an integer and either scalar or n x 1")

    last = None

    while np.any(label != last):
        u, label = np.unique(label, return_inverse=True)
        k = len(u)

        num_per_label = np.bincount(label, minlength=k)
        val_per_label = 1 / num_per_label

        E = sparse.coo_matrix(
            (val_per_label[label], (label, np.arange(n))), shape=(k, n))
        E = E.tocsr()
        T = E.dot(K)
        T_dot_ET = E.dot(T.T).T
        Z1 = np.repeat(np.diagonal(T_dot_ET).reshape(-1, 1), n, axis=1)
        Z2 = 2 * T
        Z = Z1 - Z2
        last = label.copy()
        label = np.argmin(Z, axis=0)

    val = Z[label, np.arange(n)]
    energy = np.sum(val) + np.trace(K)
    return label, energy

def repeat_kn_kmeans(K, init, num_times):
    return min((kn_kmeans(K, init) for _ in lazy_range(num_times)),
               key=op.itemgetter(1))
