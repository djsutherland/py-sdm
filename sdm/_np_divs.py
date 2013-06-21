import itertools
from functools import partial

import numpy as np
from scipy.special import gamma, gammaln

from .features import _group
from .utils import iteritems, lazy_range, izip
from .mp_utils import progress, get_pool, ForkedData
from .knn_search import knn_search

################################################################################
### Estimators of various divergences based on nearest-neighbor distances.
#
# The standard interface for these functions is:
#
# Function attributes:
#
#   needs_alpha: whether this function needs an alpha parameter. Default false.
#
#   self_value: The value that this function should take when comparing a
#               sample to itself: either a scalar constant or None (the
#               default), in which case the function is still called with
#               rhos = nus.
#
# Arguments:
#
#   alphas (if needs_alpha; array-like, scalar or 1d): the alpha values to use
#
#   Ks (array-like, scalar or 1d): the K values used
#
#   num_q (scalar): the number of points in the sample from q
#
#   dim (scalar): the dimension of the feature space
#
#   rhos: an array of within-bag nearest neighbor distances for a sample from p.
#         rhos[i, j] should be the distance from the ith sample from p to its
#         Ks[j]'th neighbor in the same sample. Shape: (num_p, num_Ks).
#   nus: an array of nearest neighbor distances from samples from other dists.
#        nus[i, j] should be the distance from the ith sample from p to its
#        Ks[j]'th neighbor in the sample from q. Shape: (num_p, num_Ks).
#
# Returns an array of divergence estimates. If needs_alpha, should be of shape
# (num_alphas, num_Ks); otherwise, of shape (num_Ks,).

def linear(Ks, num_q, dim, rhos, nus):
    r'''
    Estimates the linear inner product \int p q between two distributions,
    based on kNN distances.
    '''
    # Estimated with alpha=0, beta=1:
    #   B_{k,d,0,1} = (k - 1) / pi^(dim/2) * gamma(dim/2 + 1)
    #   (using gamma(k) / gamma(k - 1) = k - 1)
    # and the rest of the estimator is
    #   B / m * mean(nu ^ -dim)
    Ks = np.reshape(Ks, (-1,))
    Bs = (Ks - 1) / np.pi ** (dim / 2) * gamma(dim / 2 + 1)  # shape (num_Ks,)
    return Bs / num_q * np.mean(nus ** (-dim), axis=0)
linear.self_value = None  # have to execute it
linear.needs_alpha = False


def kl(Ks, num_q, dim, rhos, nus, clamp=True):
    r'''
    Estimate the KL divergence between distributions:
        \int p(x) \log (p(x) / q(x))
    using the kNN-based estimator (5) of
        Qing Wang, Sanjeev R Kulkarni, and Sergio Verdu (2009).
        Divergence Estimation for Multidimensional Densities Via
        k-Nearest-Neighbor Distances.
        IEEE Transactions on Information Theory.
        http://www.ee.princeton.edu/~verdu/reprints/WanKulVer.May2009.pdf
    which is:
        d * 1/n \sum \log (nu_k(i) / rho_k(i)) + log(m / (n - 1))

    If clamp is True (default), enforces KL >= 0.

    Returns an array of shape (num_Ks,).
    '''
    est = dim * np.mean(np.log(nus) - np.log(rhos), axis=0) + \
          np.log(num_q / (rhos.shape[0] - 1))
    if clamp:
        np.maximum(est, 0, out=est)
    return est
kl.self_value = 0
kl.needs_alpha = False


def _alpha_div_constants(alphas, Ks):
    alphas = np.reshape(alphas, (-1, 1))
    Ks = np.reshape(Ks, (1, -1))

    omas = 1 - alphas
    Bs = np.exp(gammaln(Ks) * 2 - gammaln(Ks + omas) - gammaln(Ks - omas))
    return omas, Bs, alphas

def _alpha_div(omas, Bs, alphas, Ks, num_q, dim, rhos, nus, clamp=True):
    N = rhos.shape[0]

    # the actual main estimate:
    #   rho^(- dim * est alpha) nu^(- dim * est beta)
    #   = (rho / nu) ^ (dim * (1 - alpha))
    # do some reshaping trickery to get broadcasting right
    estimates = (rhos / nus)[:, np.newaxis, :]
    estimates = estimates ** (dim * omas.reshape(1, -1, 1))
    estimates = np.mean(estimates, axis=0)  # shape (n_alphas, n_Ks)

    # We're estimating with alpha = alpha-1, beta = 1-alpha.
    # B constant in front:
    #   estimator's alpha = -beta, so volume of unit ball cancels out
    #   and then ratio of gamma functions
    # (precalculated, since it's the same for every pair we're filling in)
    estimates *= Bs

    # factors based on the sizes:
    #   1 / [ (n-1)^(est alpha) * m^(est beta) ] = ((n-1) / m) ^ (1 - alpha)
    estimates *= ((N - 1) / num_q) ** omas

    if clamp:
        np.maximum(estimates, 0, out=estimates)
    return estimates

def get_alpha_div(alphas, Ks):
    return partial(_alpha_div, *_alpha_div_constants(alphas, Ks))

def alpha_div(alphas, Ks, num_q, dim, rhos, nus, clamp=True):
    r'''
    Estimate the alpha divergence between distributions:
        \int p^\alpha q^(1-\alpha)
    based on kNN distances.

    Used in Renyi, Hellinger, Bhattacharyya, Tsallis divergences.

    The following arguments come from _alpha_div_constants:
        omas: 1 - alphas, shaped as a column vector
        Bs: the constant in the estimator based on the gamma function, of
            shape (n_alphas, n_Ks)

    If clamp is True (default), enforces that estimates are >= 0.

    Returns divergence estimates with shape (num_alphas, num_Ks).
    '''
    return get_alpha_div(alphas, Ks)(alphas, Ks, num_q, dim, rhos, nus, clamp)
alpha_div.self_value = 1
alpha_div.needs_alpha = True
alpha_div.chooser_fn = get_alpha_div


def _estimate_cross_divs(features, indices, rhos,
                         mask, funcs, specs, n_meta_only, Ks,
                         progressbar, cores):
    n_bags = len(features)
    dim = features.dim
    max_K = np.max(Ks)

    outputs = np.empty((n_bags, n_bags, len(specs) + n_meta_only, len(Ks)),
                       dtype=np.float32)
    outputs.fill(np.nan)

    # TODO: should just call functions that need self up here with rhos
    #       instead of computing nus and then throwing them out below
    any_run_self = False
    all_bags = lazy_range(n_bags)
    for func, info in iteritems(funcs):
        self_val = getattr(func, 'self_value', None)
        if self_val is not None:
            pos = np.reshape(info.pos, (-1, 1))
            outputs[all_bags, all_bags, pos, :] = self_val
        else:
            any_run_self = True

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
        #
        # TODO: Is cythonning this file/this function worth it?

        # make a boolean array of whether we want to do the ith bag
        do_bag = mask[i].copy()
        if not any_run_self:
            do_bag[i] = False

        num_q = features._n_pts[i]

        # loop over contiguous sections where do_bag is True
        change_pts = np.hstack([0, np.diff(do_bag).nonzero()[0] + 1, n_bags])
        s = int(not do_bag[0])
        for start, end in izip(change_pts[s::2], change_pts[s+1::2]):
            boundaries = features._boundaries[start:end+1]
            feats = features._features[boundaries[0]:boundaries[-1]]

            # find the nearest neighbors in features[i] from each of these bags
            nus = _group(boundaries - boundaries[0],
                         knn_search(max_K, feats, index=index)[:, Ks - 1])

            # run the base estimators using the nus on everything
            # TODO: parallelize this bit?
            for func, info in iteritems(funcs):
                pos = info.pos

                for j, nu in izip(lazy_range(start, end), nus):
                    if i == j:
                        if getattr(func, 'self_value', None) is not None:
                            continue  # already set this above
                        nu = rhos[j]  # nu counts each point as its NN...

                    outputs[j, i, pos, :] = func(Ks, num_q, dim, rhos[j], nu)
    return outputs
