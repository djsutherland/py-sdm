from __future__ import division

import numpy as np

from .utils import lazy_range, izip, iteritems
from .mp_utils import progress
from .knn_search import knn_search


def _linear(Bs, dim, num_q, rhos, nus):
    # and the rest of the estimator is
    #   B / m * mean(nu ^ -dim)
    return Bs / num_q * np.mean(nus ** (-dim), axis=0)


def kl(Ks, dim, num_q, rhos, nus):
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

    Enforces KL >= 0.

    Returns an array of shape (num_Ks,).
    '''
    est = dim * np.mean(np.log(nus) - np.log(rhos), axis=0)
    est += np.log(num_q / (rhos.shape[0] - 1))
    np.maximum(est, 0, out=est)
    return est
kl.self_value = 0
kl.needs_alpha = False


def _alpha_div(omas, Bs, dim, num_q, rhos, nus):
    N = rhos.shape[0]

    # the actual main estimate:
    #   rho^(- dim * est alpha) nu^(- dim * est beta)
    #   = (rho / nu) ^ (dim * (1 - alpha))
    # do some reshaping trickery to get broadcasting right
    estimates = (rhos / nus)[:, np.newaxis, :]
    estimates = estimates ** (dim * omas.reshape(1, -1, 1))
    estimates = np.mean(estimates, axis=0)  # shape (n_alphas, n_Ks)

    estimates *= Bs

    # factors based on the sizes:
    #   1 / [ (n-1)^(est alpha) * m^(est beta) ] = ((n-1) / m) ^ (1 - alpha)
    estimates *= ((N - 1) / num_q) ** omas

    np.maximum(estimates, 0, out=estimates)
    return estimates


################################################################################

def _estimate_cross_divs(features, indices, rhos,
                         mask, funcs, Ks, specs, n_meta_only,
                         progressbar, cores):
    n_bags = len(features)
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

        num_q = features._n_pts[i]

        # make a boolean array of whether we want to do the ith bag
        do_bag = mask[i]
        if not any_run_self:
            do_bag = do_bag.copy()
            do_bag[i] = False

        # loop over contiguous sections where do_bag is True
        change_pts = np.hstack([0, np.diff(do_bag).nonzero()[0] + 1, n_bags])
        s = 0 if do_bag[0] else 1
        for start, end in izip(change_pts[s::2], change_pts[s+1::2]):
            boundaries = features._boundaries[start:end+1]
            feats = features._features[boundaries[0]:boundaries[-1]]
            base = boundaries[0]

            # find the nearest neighbors in features[i] from each of these bags
            neighbors = knn_search(max_K, feats, index=index,
                                   min_dist=min_dist)[:, Ks - 1]

            for j_sub, j in enumerate(lazy_range(start, end)):
                rho = rhos[j]

                nu_start = boundaries[j_sub] - base
                nu_end = boundaries[j_sub + 1] - base
                nu = neighbors[nu_start:nu_end]

                if i == j:
                    for func, info in iteritems(funcs):
                        if getattr(func, 'self_value', None) is None:
                            outputs[j, i, info.pos, :] = func(num_q, rho, rho)
                            # otherwise, already set it above
                else:
                    for func, info in iteritems(funcs):
                        outputs[j, i, info.pos, :] = func(num_q, rho, nu)
    return outputs
