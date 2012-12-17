'''
Code to compute nonparametric estimates of various distance metrics and
divergences between distributions, based on kNN distances among samples.
See the README for a paper reference.
'''
from __future__ import division, absolute_import

from functools import partial

import numpy as np
from scipy.special import gamma, gammaln

from .utils import eps, col, get_col


def fix_terms(terms, tail=0.01):
    '''Removes the tails from terms, for a robust estimate of the mean.'''
    terms = terms[np.logical_not(np.isnan(terms))]
    n = terms.size

    if n >= 3:
        terms = np.sort(terms)
        terms = terms[max(1, round(n*tail)) : min(n-1, round(n*(1-tail)))]

    return terms[np.isfinite(terms)]


def symmetric(func):
    func.is_symmetric = True
    return func
def asymmetric(func):
    func.is_symmetric = False
    return func

@symmetric
def l2(xx, xy, yy, yx, Ks, dim, tail=0.05, **opts):
    '''
    Estimate the L2 distance between two divergences, based on kNN distances.
    Returns a vector: one element for each K in opt['Ks'].
    '''
    fix = partial(fix_terms, tail=tail)

    if xy is None: # identical bags
        return np.zeros(len(Ks))

    N = xx.shape[0]
    M = yy.shape[0]
    c = np.pi**(dim*0.5) / gamma(dim*0.5 + 1)

    rs = []
    for K in Ks:
        rho_x, nu_x = get_col(xx, K-1), get_col(xy, K-1)
        rho_y, nu_y = get_col(yy, K-1), get_col(yx, K-1)

        total = (fix((K-1) / ((N-1)*c) / (rho_x ** dim)).mean()
               + fix((K-1) / ((M-1)*c) / (rho_y ** dim)).mean()
               - fix((K-1) / (  M  *c) / ( nu_x ** dim)).mean()
               - fix((K-1) / (  N  *c) / ( nu_y ** dim)).mean())

        rs.append(np.sqrt(max(0, total)))

    return np.array(rs)


@asymmetric
def alpha_div(xx, xy, yy, yx, alphas, Ks, dim, tail=0.05, **opts):
    '''
    Estimate the alpha divergence between distributions, based on kNN distances.
    Used in Renyi and Hellinger divergence estimation.
    Returns a matrix: each row corresponds to an alpha, each column to a K.
    '''
    alphas = np.asarray(alphas)
    fix = partial(fix_terms, tail=tail)

    N = xx.shape[0]
    M = yy.shape[0]

    rs = np.empty((len(alphas), len(Ks)))
    for knd, K in enumerate(Ks):
        K = Ks[knd]

        rho, nu = get_col(xx, K-1), get_col(xy, K-1)
        ratios = fix(rho / nu)

        for ind, alpha in enumerate(alphas):
            es = (((N-1) / M) ** (1-alpha)) * (ratios ** (dim*(1-alpha))).mean()
            B = np.exp(gammaln(K)*2 - gammaln(K+1-alpha) - gammaln(K+alpha-1))

            rs[ind, knd] = es * B

    return rs

@asymmetric
def renyi(xx, xy, yy, yx, alphas, Ks, **opts):
    '''
    Estimate the Renyi-alpha divergence between distributions, based on kNN
    distances.
    Returns a matrix: each row corresponds to an alpha, each column to a K.
    '''
    alphas = np.asarray(alphas)
    Ks = np.asarray(Ks)

    if xy is None: # identical bags
        return np.zeros((len(alphas), len(Ks)))

    alphas[alphas == 1] = 0.99 # approximate KL
    est = alpha_div(xx, xy, yy, yx, alphas=alphas, Ks=Ks, **opts)
    return np.maximum(0, np.log(np.maximum(est, eps)) / (col(alphas) - 1))


@asymmetric # the metric is symmetric, but estimator is not
def hellinger(xx, xy, yy, yx, Ks, dim, tail=0.05, **opts):
    '''
    Estimate the Hellinger distance between distributions, based on kNN
    distances.
    Returns a vector: one element for each K.
    '''
    if xy is None: # identical bags
        return np.zeros(len(Ks))

    est = alpha_div(xx, xy, yy, yx, alphas=[0.5], Ks=Ks, dim=dim, tail=tail)
    return np.sqrt(np.maximum(0, 1 - est))
