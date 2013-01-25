#!/usr/bin/env python
'''
Script to run nonparametric divergence estimation via k-nearest-neighbor
distances. Based on the method of
    Barnabas Poczos, Liang Xiong, Jeff Schneider (2011).
    Nonparametric divergence estimation with applications to machine learning
    on distributions.
    Uncertainty in Artificial Intelligence.
    http://autonlab.org/autonweb/20287.html
'''

from __future__ import division, print_function

import os
assert os.name == 'posix', 'the os should support fork()'

import argparse
import functools
import itertools
import re
import sys
import warnings

import bottleneck as bn
import h5py
import numpy as np
import scipy.io
from scipy.special import gamma, gammaln

try:
    from pyflann import FLANN
    searcher = FLANN()
except ImportError:
    warnings.warn('Cannot find FLANN. KNN searches will be much slower.')
    searcher = None

from utils import (eps, col, get_col, izip, lazy_range, is_integer, raw_input,
                   str_types, portion, positive_int, confirm_outfile)
from mp_utils import ForkedData, map_unordered_with_progressbar, get_pool

################################################################################
### Helpers for robust mean estimation
# XXX: old code used _clip, but _trim seems better

TAIL_DEFAULT = 0.01
FIX_MODE_DEFAULT = 'trim'

def fix_terms_trim(terms, tail=TAIL_DEFAULT):
    '''
    Trims the elements of an array, to use in a more robust mean estimate, by:
        - removing any nan elements
        - removing elements below the tail-th and above the (1-tail)th quantiles
        - removing any remaining inf elements
    '''
    terms = terms[np.logical_not(np.isnan(terms))]
    n = terms.size

    if n >= 3:
        terms = np.sort(terms)  # TODO: could do this with partial sorting
        ends = int(round(n * tail))
        terms = terms[ends:n-ends]

    return terms[np.isfinite(terms)]

def quantile(a, prob):
    '''
    Estimates the prob'th quantile of the values in a data array.

    Uses the algorithm of matlab's quantile(), namely:
        - Remove any nan values
        - Take the sorted data as the (.5/n), (1.5/n), ..., (1-.5/n) quantiles.
        - Use linear interpolation for values between (.5/n) and (1 - .5/n).
        - Use the minimum or maximum for quantiles outside that range.

    See also: scipy.stats.mstats.mquantiles
    '''
    a = np.asanyarray(a)
    a = a[np.logical_not(np.isnan(a))].ravel()
    n = a.size

    if prob >= 1 - .5/n:
        return a.max()
    elif prob <= .5 / n:
        return a.min()

    # find the two bounds we're interpreting between:
    # that is, find i such that (i+.5) / n <= prob <= (i+1.5)/n
    t = n * prob - .5
    i = int(np.floor(t))

    # partial sort so that the ith element is at position i, with bigger ones
    # to the right and smaller to the left
    a = bn.partsort(a, i)

    if i == t:  # did we luck out and get an integer index?
        return a[i]
    else:
        # we'll linearly interpolate between this and the next index
        smaller = a[i]
        larger = a[i+1:].min()
        if np.isinf(smaller):
            return smaller  # avoid inf - inf
        else:
            return smaller + (larger - smaller) * (t - i)


def fix_terms_clip(terms, tail=TAIL_DEFAULT):
    '''
    Takes a vector of elements and replaces any infinite or very-large elements
    with the value of the highest non-very-large element, as well as throwing
    away any nan values, possibly changing the order.

    Used for estimating the mean of positive quantities.

    "Very-large" is defined as the (1-tail)th quantile if tail > 0, otherwise
    the largest non-inf element. Note that values of -inf are not altered.

    Uses the matlab-style quantile() function, above, because that's what the
    code this is trying to replicate did.
    '''
    terms = terms[np.logical_not(np.isnan(terms))]

    find_noninf_max = True
    if tail > 0:
        cutoff = quantile(terms, 1 - tail)
        find_noninf_max = not np.isfinite(cutoff)

    if find_noninf_max:
        cutoff = np.max(terms[np.isfinite(terms)])

    terms[terms > cutoff] = cutoff
    return terms

_fix_term_modes = {'trim': fix_terms_trim, 'clip': fix_terms_clip}
def fix_terms(terms, tail=TAIL_DEFAULT, mode=FIX_MODE_DEFAULT):
    return _fix_term_modes[mode](terms, tail=tail)


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

def default_min_dist(dim):
    return min(1e-2, 1e-100 ** (1.0 / dim))

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
        min_dist = default_min_dist(dim)
    elif min_dist <= 0:
        return dist, idx
    return np.maximum(min_dist, dist), idx


def knn_search_forked(bags, i, j, K, min_dist=None):
    bags = bags.value
    return knn_search(bags[i], bags[j], K, min_dist=min_dist)


################################################################################
### Estimators of various divergences based on nearest-neighbor distances.

def l2(xx, xy, yy, yx, Ks, dim, tail=TAIL_DEFAULT, fix_mode=FIX_MODE_DEFAULT,
       **opts):
    '''
    Estimate the L2 distance between two divergences, based on kNN distances.
    Returns a vector: one element for each K in opt['Ks'].
    '''
    fix = functools.partial(fix_terms, tail=tail, mode=fix_mode)

    if xy is None:  # identical bags
        return np.zeros(len(Ks))

    N = xx.shape[0]
    M = yy.shape[0]
    c = np.pi ** (dim * 0.5) / gamma(dim * 0.5 + 1)

    rs = []
    for K in Ks:
        rho_x, nu_x = get_col(xx, K-1), get_col(xy, K-1)
        rho_y, nu_y = get_col(yy, K-1), get_col(yx, K-1)

        total = (fix((K-1) / ((N-1)*c) / (rho_x ** dim)).mean()
               + fix((K-1) / ((M-1)*c) / (rho_y ** dim)).mean()
               - fix((K-1) / (  M * c) / ( nu_x ** dim)).mean()
               - fix((K-1) / (  N * c) / ( nu_y ** dim)).mean())

        rs.append(np.sqrt(max(0, total)))

    return np.array(rs)
l2.is_symmetric = True
l2.name = 'NP-L2'


def alpha_div(xx, xy, yy, yx, alphas, Ks, dim,
              tail=TAIL_DEFAULT, fix_mode=FIX_MODE_DEFAULT, **opts):
    '''
    Estimate the alpha divergence between distributions, based on kNN distances.
    Used in Renyi and Hellinger divergence estimation.
    Returns a matrix: each row corresponds to an alpha, each column to a K.
    '''
    if xy is None:  # identical bags
        return np.ones((len(alphas), len(Ks)))

    alphas = np.asarray(alphas)
    fix = functools.partial(fix_terms, tail=tail, mode=fix_mode)

    N = xx.shape[0]
    M = yy.shape[0]

    rs = np.empty((len(alphas), len(Ks)))
    for knd, K in enumerate(Ks):
        K = Ks[knd]

        rho, nu = get_col(xx, K - 1), get_col(xy, K - 1)
        ratios = fix(rho / nu)

        for ind, alpha in enumerate(alphas):
            es = (((N-1) / M) ** (1-alpha)) * (ratios ** (dim*(1-alpha))).mean()
            B = np.exp(gammaln(K)*2 - gammaln(K+1-alpha) - gammaln(K+alpha-1))

            rs[ind, knd] = es * B

    return rs
alpha_div.is_symmetric = False
alpha_div.name = 'NP-A'


def bhattacharyya(xx, xy, yy, yx, Ks, **opts):
    del opts['alphas']
    return alpha_div(xx, xy, yy, yx, alphas=[.5], Ks=Ks, **opts)
bhattacharyya.is_symmetric = False
bhattacharyya.name = 'NP-BC'


def renyi(xx, xy, yy, yx, alphas, Ks, **opts):
    '''
    Estimate the Renyi-alpha divergence between distributions, based on kNN
    distances.
    Returns a matrix: each row corresponds to an alpha, each column to a K.
    '''
    alphas = np.asarray(alphas)
    Ks = np.asarray(Ks)

    alphas[alphas == 1] = 0.99  # approximate KL
    est = alpha_div(xx, xy, yy, yx, alphas=alphas, Ks=Ks, **opts)
    return np.maximum(0, np.log(np.maximum(est, eps)) / (col(alphas) - 1))
renyi.is_symmetric = False
renyi.name = 'NP-R'


def hellinger(xx, xy, yy, yx, Ks, dim,
              tail=TAIL_DEFAULT, fix_mode=FIX_MODE_DEFAULT, **opts):
    '''
    Estimate the Hellinger distance between distributions, based on kNN
    distances.
    Returns a vector: one element for each K.
    '''
    est = np.squeeze(alpha_div(xx, xy, yy, yx, alphas=[0.5], Ks=Ks, dim=dim,
                               tail=tail, fix_mode=fix_mode))
    return np.sqrt(np.maximum(0, 1 - est))
hellinger.is_symmetric = False  # the metric is symmetric, but estimator is not
hellinger.name = 'NP-H'

func_mapping = {
    'l2': l2,
    'alpha': alpha_div,
    'bc': bhattacharyya,
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


def process_pair(funcs, opts, bags, xxs, row, col, symm=True):
    '''
    Does nearest-neighbor searches and computes the divergences between
    the row-th and col-th bags. xxs is a list of nearest-neighbor searches
    within each bag.
    '''
    handle = functools.partial(handle_divs, funcs, opts)

    bags = bags.value
    xxs = xxs.value

    if row == col:
        xx = xxs[row]
        r = handle(xx, (None, None), (None, None), (None, None))
        if symm:
            rt = r
    else:
        xbag, ybag = bags[row], bags[col]
        xx, yy = xxs[row], xxs[col]

        mK = max(opts['Ks'])
        xy = knn_search(xbag, ybag, mK)
        yx = knn_search(ybag, xbag, mK)

        r = handle(xx, xy, yy, yx)
        if symm:
            rt = handle(yy, yx, xx, xy, r)

    r = np.hstack([np.asarray(a, dtype='float32').flat for a in r])
    if symm:
        rt = np.hstack([np.asarray(a, dtype='float32').flat for a in rt])

    return row, col, r, (rt if symm else None)


################################################################################
### Argument handling.

def process_func_specs(div_specs, Ks):
    alphas = []
    funcs = []
    div_names = []
    for func_spec in div_specs:
        if ':' in func_spec:
            func_name, alpha_spec = func_spec.split(':', 2)

            new_alphas = np.sort([float(a) for a in alpha_spec.split(',')])
            if alphas and not np.all(alphas == new_alphas):
                raise ValueError("Can't do conflicting alpha options yet.")
            alphas = new_alphas

            func = func_mapping[func_name]
            funcs.append(func)
            div_names.extend(['{}[a={},K={}]'.format(func.name, al, K)
                              for al in alphas for K in Ks])
        else:
            func = func_mapping[func_spec]
            funcs.append(func)
            div_names.extend(['{}[K={}]'.format(func.name, K)
                              for K in Ks])
    return funcs, div_names, alphas

def read_cell_array(f, data):
    return [
        np.ascontiguousarray(np.transpose(f[ptr]), dtype=np.float32)
        for row in data
        for ptr in row
    ]

def read_data(input_file, input_var, n_points=None):
    with h5py.File(input_file, 'r') as f:
        return read_cell_array(f, f[input_var], n_points)

def subset_data(bags, n_points):
    return [x[np.random.permutation(x.shape[0])[:n_points]]
            if x.shape[0] < n_points else x for x in bags]

################################################################################
### The main dealio

def get_divs(bags,
             mask=None,
             specs=['renyi:.9'],
             Ks=[3],
             n_proc=None,
             tail=TAIL_DEFAULT,
             fix_mode=FIX_MODE_DEFAULT,
             min_dist=None,
             status_fn=True, progressbar=None,
             return_opts=False):
    '''
    Gets the divergences between bags.
        bags: a length n list of row-instance feature matrices
        mask: an n x n boolean array: whether to estimate each div pair
        specs: a list of strings of divergence specs
        Ks: a K values
        n_proc: number of processes to use; None for # of cores
        tail: an argument for fix_terms (above)
        fix_mode: mode arg for fix_terms, above
        min_dist: a minimum distance to use in kNN searches
        status_fn: a function to print out status messages
            None means don't print any; True prints to stderr
        progressbar: show a progress bar on stderr. default: (status_fn is True)
        return_opts: return a dictionary of options used as second value
    Returns a matrix of size  n x n x num_div_funcs
    '''
    # TODO: document progressbar specs
    # TODO: other kinds of callbacks for showing progress bars

    if progressbar is None:
        progressbar = status_fn is True

    opts = {}
    opts['Ks'] = np.sort(np.ravel(Ks))
    opts['tail'] = tail
    opts['fix_mode'] = fix_mode
    opts['min_dist'] = min_dist
    funcs, opts['div_names'], opts['alphas'] = \
            process_func_specs(specs, opts['Ks'])

    if status_fn is True:
        status_fn = functools.partial(print, file=sys.stderr)
    elif status_fn is None:
        status_fn = lambda *args, **kwargs: None

    num_bags = len(bags)
    opts['dim'] = dim = bags[0].shape[1]
    assert all(bag.shape[1] == dim for bag in bags)
    max_K = opts['Ks'][-1]

    if mask is not None:
        assert mask.dtype == np.dtype('bool')
        assert mask.shape == (num_bags, num_bags)

    status_fn('kNN processing: {} bags, dimension {}, # points {} to {}, K = {}'
            .format(num_bags, dim,
                    min(b.shape[0] for b in bags),
                    max(b.shape[0] for b in bags),
                    max_K))

    status_fn('Preparing...')
    bag_data = ForkedData(bags)

    # do kNN searches within each bag
    # need to throw away the closest neighbor, which will always be self
    knn_searcher = functools.partial(knn_search_forked,
            bag_data, K=max_K + 1, min_dist=min_dist)
    bag_is = lazy_range(num_bags)
    with get_pool(n_proc) as pool:
        xxs = [(xx[:, 1:], xxi[:, 1:]) for xx, xxi
                in pool.starmap(knn_searcher, izip(bag_is, bag_is))]

    xxs_data = ForkedData(xxs)

    processor = functools.partial(process_pair, funcs, opts, bag_data, xxs_data)
    all_pairs = itertools.combinations_with_replacement(range(num_bags), 2)
    if mask is None:
        jobs = list(all_pairs)
    else:
        jobs = []  # probably a nicer way to do this...
        for i, j in all_pairs:
            if mask[i, j]:
                jobs.append((i, j, mask[j, i]))
            elif mask[j, i]:
                jobs.append((j, i, False))

    status_fn('Doing the real work...')
    with get_pool(n_proc) as pool:
        if progressbar:
            rs = map_unordered_with_progressbar(pool, processor, jobs)
        else:
            rs = pool.starmap(processor, jobs)

    R = np.empty((num_bags, num_bags, rs[0][2].size), dtype=np.float32)
    R.fill(np.nan)
    for i, j, r, rt in rs:
        R[i, j, :] = r
        if rt is not None:
            R[j, i, :] = rt

    return (R, opts) if return_opts else R


################################################################################
### Command line interface

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute divergences and set kernels based on "
                    "KNN statistics.")

    parser.add_argument('input_file',
        help="The input file, an HDF5 file (e.g. .mat with -v7.3).")
    parser.add_argument('--input-format',
        choices=['matlab', 'python'], default='python',
        help="Whether the features file was generated by the matlab code or "
             "the python code; default python.")
    parser.add_argument('--input-var-name',
        help="The name of the cell array of row-instance data matrices, "
             "if the input file is matlab format.")

    parser.add_argument('output_file', nargs='?',
        help="Name of the output file; default input_file.py_divs.(mat|h5).")
    parser.add_argument('--output-format',
        choices=['hdf5', 'mat'], default='hdf5',
        help="Output file format; default %(default)s.")

    parser.add_argument('--n-proc', type=positive_int, default=None,
        help="Number of processes to use; default is as many as CPU cores.")
    parser.add_argument('--n-points', type=positive_int, default=None,
        help="The number of points to use per group; defaults to all.")

    parser.add_argument('--div-funcs', nargs='*',
        default=['hellinger', 'l2', 'renyi:.5,.7,.9,.99'],
        help="The divergences to estimate. Default: %(default)s.")

    parser.add_argument('-K', nargs='*', type=positive_int,
        default=[1, 3, 5, 10],
        help="The numbers of nearest neighbors to calculate.")

    parser.add_argument('--trim-tails', type=portion, default=TAIL_DEFAULT,
        help="How much to trim off the ends of things we take the mean of; "
             "default %(default)s.", metavar='PORTION')
    parser.add_argument('--trim-mode', choices=['clip', 'trim'],
        default=FIX_MODE_DEFAULT,
        help="Whether to trim or clip ends; default %(default)s.")

    parser.add_argument('--min-dist', type=float, default=None,
        help="Protect against identical points by making sure kNN distances "
             "are always at least this big. Default: the smaller of .01 and "
             "10 ^ (100 / dim).")

    # TODO: FLANN nearest-neighbor algorithm selection

    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = '{}.py_divs.{}'.format(
            args.input_file, 'mat' if args.output_format == 'mat' else 'h5')

    return args


def main():
    args = parse_args()
    status_fn = functools.partial(print, file=sys.stderr)

    status_fn('Reading data...')
    if args.input_format == 'matlab':
        data = None
        bags = read_data(args.input_file, args.input_var_name)
    else:
        from extract_features import read_features
        data = read_features(args.input_file)
        bags = data.features

    dim = bags[0].shape[1]
    if args.min_dist is None:
        args.min_dist = default_min_dist(dim)

    if args.output_format == 'mat':
        confirm_outfile(args.output_file)
    else:
        if not os.path.exists(args.output_file):
            confirm_outfile(args.output_file)
        else:
            check_h5_file_agreement(args.output_file, bags, data, args)
            status_fn("Output file already exists, but agrees with args.")

    if args.n_points:
        bags = subset_data(bags, args.n_points)

    R, opts = get_divs(
            bags, specs=args.div_funcs, Ks=args.K,
            n_proc=args.n_proc,
            tail=args.trim_tails, fix_mode=args.trim_mode,
            min_dist=args.min_dist,
            return_opts=True)

    status_fn("Outputting results to", args.output_file)

    opts['Ds'] = R
    if opts['min_dist'] is None:
        opts['min_dist'] = default_min_dist(opts['dim'])

    if data is not None:
        opts['labels'] = data.labels
        opts['names'] = data.names

    if args.output_format == 'mat':
        scipy.io.savemat(args.output_file, opts, oned_as='column')
    else:
        add_to_h5_file(args.output_file, opts)

    assert not np.any(np.isnan(R)), 'nan found in the result'
    assert not np.any(np.isinf(R)), 'inf found in the result'


def check_h5_file_agreement(filename, bags, data, args):
    with h5py.File(filename) as f:
        # output file already exists; make sure args agree
        if not f.attrs.keys() and not f.keys():
            return
        assert np.all(f.attrs['fix_mode'] == args.trim_mode)
        assert np.all(f.attrs['tail'][...] == args.trim_tails)
        assert np.all(f.attrs['min_dist'][...] == args.min_dist)
        if 'names' in f.attrs and data is not None:
            assert np.all(f.attrs['names'] == data.names)
        if 'labels' in f.attrs and data is not None:
            assert np.all(f.attrs['labels'] == data.labels)

        n = len(bags)
        assert all(divs.shape == (n, n)
                   for div_group in f.values()
                   for divs in div_group.values())

        # any overlap with stuff we've already calculated?
        div_funcs = []
        for div_func in args.div_funcs:
            div_funcs.extend(normalize_div_name(div_func))
        overlap = [(div_func, k)
                   for div_func in div_funcs if div_func in f
                   for k in args.K if str(k) in f[div_func]]
        if overlap:
            msg = '\n'.join(
                ["WARNING: the following divs will be overwritten:"] +
                ['\t{}, k = {}'.format(df, k) for df, k in overlap] +
                ['Proceed? [yN] '])
            resp = raw_input(msg)
            if not resp.startswith('y'):
                sys.exit("Aborting.")


def add_to_h5_file(filename, opts):
    import h5py
    with h5py.File(filename) as f:
        for div_name, divs in izip(opts['div_names'],
                                   np.rollaxis(opts['Ds'], axis=-1)):
            name, k = reverse_div_name(div_name)
            g = f.require_group(name)
            if k in g:
                g[k][...] = divs
            else:
                g[k] = divs

        for k, v in opts.items():
            if k in ('Ds', 'Ks', 'alphas', 'div_names'):
                continue
            if isinstance(v, list) and isinstance(v[0], str_types):
                v = np.asarray(v, dtype=h5py.special_dtype(vlen=str))
            f.attrs[k] = v


_rev_name_map = dict((f.name, k) for k, f in func_mapping.items())
_name_fmt = re.compile(r'''
    ([^\[]+)          # div func name, eg NP-H
    \[
    (?:a=([.\d]+),)?  # optional alpha specification
    K=(\d+)           # K spec
    \]
''', re.VERBOSE)
def reverse_div_name(name):
    '''
    Parse names like NP-H[K=1] or NP-R[a=0.9,K=5]
    into something like ('hellinger', 1) or ('renyi:.9', 5).
    '''
    # TODO: make the reversal unnecessary...sigh
    div_name, alpha, k = _name_fmt.match(name).groups()
    div_name = _rev_name_map[div_name]
    s = '{}:{}'.format(div_name, alpha) if alpha else div_name
    return s, k

def normalize_div_name(name):
    if ':' in name:
        main, alpha = name.split(':')
        return ['{}:{}'.format(main, float(al))
                for al in alpha.split(',')]
    return [name]

if __name__ == '__main__':
    main()
