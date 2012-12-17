#!/usr/bin/env python
'''
Script to run nonparametric divergence estimation via k-nearest-neighbor
distances. Based on the method of
    Barnabas Poczos, Liang Xiong, Jeff Schneider (2011).
    Nonparametric divergence estimation with applications to machine learning on distributions.
    Uncertainty in Artificial Intelligence.
    http://autonlab.org/autonweb/20287.html

Code by Liang Xiong and Dougal J. Sutherland - {lxiong,dsutherl}@cs.cmu.edu

The main estimator part only works when run as __main__; it can't be called
directly from other code, because it needs to do gross things with global
variables to pass data to fork()ed processes without having to pickle it.
For this reason it also does not work on Windows.

Written for Python 2.7 but with Python 3.2+ compatability in mind; currently
untested on 3.x because I only have 3.3 installed, and h5py is still broken
there. Uses some functionality not included in 2.6, but replacing the
itertools.combinations_with_replacement() and installing the argparse package
should probably be enough to get that to work if you need it to.
'''

from __future__ import division, print_function

import os
assert os.name == 'posix', 'the os should support fork()'

import argparse
import functools
import itertools
import multiprocessing as mp
import warnings
import sys

import h5py
import numpy as np
import progressbar as pb
import scipy.io
from scipy.special import gamma, gammaln

try:
    from pyflann import FLANN
    searcher = FLANN()
except ImportError:
    warnings.warn('Cannot find FLANN. KNN searches will be much slower.')
    searcher = None


if sys.version_info.major == 2:
    izip = itertools.izip
    imap = itertools.imap
    strict_map = map
else:
    izip = zip
    imap = map
    @functools.wraps(map)
    def strict_map(*args, **kwargs):
        return list(map(*args, **kwargs))


################################################################################
### Various small utilities.

eps = np.spacing(1)

def col(a): return a.reshape((-1, 1))
def row(a): return a.reshape((1, -1))
def get_col(X, c): return X[:,c].ravel()

def is_integer(x):
    return np.isscalar(x) and \
            issubclass(np.asanyarray(x).dtype.type, np.integer)

TAIL_DEFAULT = 0.01
def fix_terms(terms, tail=TAIL_DEFAULT):
    '''Removes the tails from terms, for a more robust estimate of the mean.'''
    terms = terms[np.logical_not(np.isnan(terms))]
    n = terms.size

    if n >= 3:
        terms = np.sort(terms)
        terms = terms[max(1, round(n*tail)) : min(n-1, round(n*(1-tail)))]

    return terms[np.isfinite(terms)]


################################################################################
### Nearest neighbor searches.

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


################################################################################
### Estimators of various divergences based on nearest-neighbor distances.

def l2(xx, xy, yy, yx, Ks, dim, tail=TAIL_DEFAULT, **opts):
    '''
    Estimate the L2 distance between two divergences, based on kNN distances.
    Returns a vector: one element for each K in opt['Ks'].
    '''
    fix = functools.partial(fix_terms, tail=tail)

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
l2.is_symmetric = True
l2.name = 'NP-L2'

def alpha_div(xx, xy, yy, yx, alphas, Ks, dim, tail=TAIL_DEFAULT, **opts):
    '''
    Estimate the alpha divergence between distributions, based on kNN distances.
    Used in Renyi and Hellinger divergence estimation.
    Returns a matrix: each row corresponds to an alpha, each column to a K.
    '''
    alphas = np.asarray(alphas)
    fix = functools.partial(fix_terms, tail=tail)

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
alpha_div.is_symmetric = False
alpha_div.name = 'NP-A'

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
renyi.is_symmetric = False
renyi.name = 'NP-R'


def hellinger(xx, xy, yy, yx, Ks, dim, tail=TAIL_DEFAULT, **opts):
    '''
    Estimate the Hellinger distance between distributions, based on kNN
    distances.
    Returns a vector: one element for each K.
    '''
    if xy is None: # identical bags
        return np.zeros(len(Ks))

    est = alpha_div(xx, xy, yy, yx, alphas=[0.5], Ks=Ks, dim=dim, tail=tail)
    return np.sqrt(np.maximum(0, 1 - est))
hellinger.is_symmetric = False # the metric is symmetric, but estimator is not
hellinger.name = 'NP-H'

func_mapping = {
    'l2': l2,
    'alpha': alpha_div,
    'renyi': renyi,
    'hellinger': hellinger,
}


################################################################################
### Parallelization helpers for computing estimators.

def handle_divs(funcs, opts, xx, xy, yy, yx, rt=None):
    '''
    Processes a set of pairwise nearest-neighbor distances into
    divergence estimates.

    funcs: a list of functions from div_estimation
    xx: nearest-neighbor distances from x to x
    xy: nearest-neighbor distances from x to y
    yx: nearest-neighbor distances from y to x
    yy: nearest-neighbor distances from y to y
    rt: the results from the symmetric call
    '''

    return [
        symm_result if (f.is_symmetric and symm_result is not None)
        else np.asarray(f(xx[0], xy[0], yy[0], yx[0], **opts))
        for f, symm_result in izip(funcs, rt or itertools.repeat(None))
    ]
    # NOTE: if we add div funcs that need the index, pass those.


# Will store the global data.
bags = []
xxs = []

def process_pair(funcs, opts, row, col):
    handle = functools.partial(handle_divs, funcs, opts)

    if row == col: # XXX support searches from X to Y
        xx = xxs[row]
        r = handle(xx, (None,None), (None,None), (None,None))
        rt = r
    else:
        xbag, ybag = bags[row], bags[col]
        xx, yy = xxs[row], xxs[col]

        mK = max(opts['Ks'])
        xy = knn_search(xbag, ybag, mK)
        yx = knn_search(ybag, xbag, mK)

        r = handle(xx, xy, yy, yx)
        rt = handle(yy, yx, xx, xy, r)

    # XXX what's going on here?
    for ind in range(len(r)):
        r[ind] = np.asarray(r[ind], dtype='float32').ravel()
        rt[ind] = np.asarray(rt[ind], dtype='float32').ravel()

    return row, col, np.hstack(r), np.hstack(rt)


################################################################################
### Argument handling.

def parse_args():
    parser = argparse.ArgumentParser(description=
            "Compute divergences and set kernels based on KNN statistics.")

    parser.add_argument('input_mat_file',
        help="The input file, an HDF5 file (e.g. .mat with -v7.3).")
    parser.add_argument('var_name',
        help="The name of the cell array of row-instance data matrices.")
    parser.add_argument('output_mat_file', nargs='?',
        help="Name of the output file; defaults to input_mat_file.py_divs.mat.")

    parser.add_argument('--n-proc', type=int, default=None,
        help="Number of processes to use; default is as many as CPU cores.")
    parser.add_argument('--n-points', type=int, default=None,
        help="The number of points to use per group; defaults to all.")

    parser.add_argument('--div-funcs', nargs='*',
        default=['hellinger', 'l2', 'renyi:.5,.7,.9,1'], # XXX .99'],
        help="The divergences to estimate. Default: %(default)s.")

    parser.add_argument('-K', nargs='*', type=int, default=[1,3,5,10],
        help="The numbers of nearest neighbors to calculate.")

    parser.add_argument('--trim-tails', type=float, default=TAIL_DEFAULT,
        help="How much to trim off the ends of things we take the mean of; "
             "default %(default)s.", metavar='PORTION')

    # TODO: FLANN nearest-neighbor algorithm selection

    args = parser.parse_args()
    if args.output_mat_file is None:
        args.output_mat_file = args.input_mat_file + '.py_divs.mat'

    return args


def read_data(input_file, input_var, n_points=0):
    "Reads input data into the global variable 'bags'."
    with h5py.File(input_file, 'r') as f:
        for row in f[input_var]:
            for ptr in row:
                x = np.asarray(f[ptr])
                x = np.ascontiguousarray(x.T, dtype=np.float32)
                if n_points and n_points < x.shape[0]:
                    x = x[np.random.permutation(x.shape[0] - 1)[:n_points]]
                bags.append(x) # add to global variable

################################################################################
### Convenience stuff related to multiprocessing.Pool

def _apply(func_args):
    func, args = func_args
    return func(*args)

def patch_starmap(pool):
    '''
    A function that adds the equivalent of multiprocessing.Pool.starmap
    to a given pool if it doesn't have the function.
    '''
    if hasattr(pool, 'starmap'):
        return

    def starmap(func, *iterables):
        return pool.map(_apply, izip(itertools.repeat(func), izip(*iterables)))
    pool.starmap = starmap

def make_pool(n_proc):
    if n_proc != 1:
        pool = mp.Pool(n_proc)
    else:
        class ImmediateResult(object):
            def __init__(self, value): self.value = value
            def get(self, timeout=None): return self.value
            def wait(self, timeout=None): pass
            def ready(self): return True
            def successful(self): return True

        class DummyPool(object):
            def close(self): pass

            def apply_async(self, func, args, kwds=None, callback=None):
                val = func(*args, **(kwds or {}))
                callback(val)
                return ImmediateResult(val)

            def map(self, func, args): return strict_map(func, args)
            def imap_unordered(self, func, args): return imap(func, args)

        pool = DummyPool()

    patch_starmap(pool)
    return pool

################################################################################
### Progress-bar handling with multiprocessing pools

def progress(counter=True, **kwargs):
    try:
        widgets = kwargs.pop('widgets')
    except KeyError:
        if counter:
            widgets = [pb.SimpleProgress(), ' (', pb.Percentage(), ') ']
        else:
            widgets = [pb.Percentage(), ' ']
        widgets.extend((pb.Bar(), ' ', pb.ETA()))
    return pb.ProgressBar(widgets=widgets, **kwargs)

def progressbar_and_updater(*args, **kwargs):
    pbar = progress(*args, **kwargs).start()
    counter = itertools.count(1)
    def update_pbar():
        pbar.update(next(counter))
        # race conditions mean the pbar might be updated backwards on
        # occasion, but the highest count ever seen will be right.
    return pbar, update_pbar

def map_unordered_with_progressbar(pool, func, jobs):
    pbar, tick_pbar = progressbar_and_updater(maxval=len(jobs))
    def callback(result):
        tick_pbar()

    results = [pool.apply_async(func, job, callback=callback) for job in jobs]
    values = [r.get() for r in results]
    pbar.finish()
    return values

################################################################################
### The main dealio

if __name__ == '__main__':
    args = parse_args()

    # TODO: allow different options for different functions
    opts = {}
    opts['Ks'] = np.sort(args.K)
    opts['tail'] = args.trim_tails
    opts['alphas'] = []

    funcs = []
    div_names = []
    for func_spec in args.div_funcs:
        if ':' in func_spec:
            func_name, alpha_spec = func_spec.split(':', 2)

            alphas = np.sort([float(a) for a in alpha_spec.split(',')])
            if opts['alphas'] and not np.all(opts['alphas'] == alphas):
                raise ValueError("Can't do conflicting alpha options yet.")
            opts['alphas'] = alphas

            func = func_mapping[func_name]
            funcs.append(func)
            div_names.extend(['{}[a={},K={}]'.format(func.name, al, K)
                              for al in opts['alphas'] for K in opts['Ks']])
        else:
            func = func_mapping[func_spec]
            funcs.append(func)
            div_names.extend(['{}[K={}]'.format(func.name, K)
                              for K in opts['Ks']])

    status = functools.partial(print, file=sys.stderr)

    status('Reading data...')
    read_data(args.input_mat_file, args.var_name, args.n_points)

    num_bags = len(bags)
    opts['dim'] = dim = bags[0].shape[1]
    assert all(bag.shape[1] == dim for bag in bags)
    max_K = max(args.K)

    status('kNN processing: {} bags, dimension {}, # points {} to {}, K = {}'
            .format(num_bags, dim,
                    min(b.shape[0] for b in bags),
                    max(b.shape[0] for b in bags),
                    max_K))

    status('Preparing...')
    pool = make_pool(args.n_proc)

    # x-to-x search needs to throw away closest neighbor of self for each pt
    xxs = pool.starmap(knn_search, bags, bags, itertools.repeat(max_K+1))
    xxs = [(xx[:,1:], xxi[:,1:]) for xx, xxi in xxs]

    # put xxs into the forked space
    pool.close()
    pool = make_pool(args.n_proc)

    jobs = list(itertools.combinations_with_replacement(range(num_bags), 2))
    processor = functools.partial(process_pair, funcs, opts)

    status('Starting for real...')
    rs = map_unordered_with_progressbar(pool, processor, jobs)

    R = np.empty((num_bags, num_bags, rs[0][2].size), dtype=np.float32)
    R.fill(np.nan)
    for i, j, r, rt in rs:
        R[i, j, :] = r
        R[j, i, :] = rt

    status("Outputting results to", args.output_mat_file)
    opts['Ds'] = R
    opts['div_names'] = div_names
    scipy.io.savemat(args.output_mat_file, opts, oned_as='column')

    assert not np.any(np.isnan(R)), 'nan found in the result'
    assert not np.any(np.isinf(R)), 'inf found in the result'
