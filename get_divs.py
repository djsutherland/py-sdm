#!/usr/bin/env python
'''
Interface to run nonparametric divergence estimation.

This is intended to be run as a script, not imported from other code,
and doesn't work on Windows. This is so it can pass data to fork()ed processes
with having to pickle it and send it through interprocess communication.
'''

from __future__ import division, print_function, absolute_import

import os
assert os.name == 'posix', 'the os should support fork()'

import argparse
import functools
import itertools
import multiprocessing as mp
import sys

import h5py
import numpy as np
import progressbar as pb
import scipy.io

from sdmpy.knn import knn_search
import sdmpy.div_estimation as div_est

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
        help="The divergences to estimate.")

    parser.add_argument('-K', nargs='*', type=int, default=[1,3,5,10],
        help="The numbers of nearest neighbors to calculate.")

    # TODO: FLANN nearest-neighbor algorithm selection

    args = parser.parse_args()
    if args.output_mat_file is None:
        args.output_mat_file = args.input_mat_file + '.py_divs.mat'

    return args


def read_data(input_file, input_var, n_points=0):
    with h5py.File(input_file, 'r') as f:
        for row in f[input_var]:
            for ptr in row:
                x = np.asarray(f[ptr])
                x = np.ascontiguousarray(x.T, dtype=np.float32)
                if n_points and n_points < x.shape[0]:
                    x = x[np.random.permutation(x.shape[0] - 1)[:n_points]]
                bags.append(x) # add to global variable

################################################################################

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

func_mapping = {
    'hellinger': div_est.hellinger,
    'renyi': div_est.renyi,
    'l2': div_est.l2,
}
name_mapping = {
    'hellinger': 'NP-H',
    'renyi': 'NP-R',
    'l2': 'NP-L2',
}


if __name__ == '__main__':
    args = parse_args()

    # TODO: allow different options for different functions
    opts = {}
    opts['Ks'] = np.sort(args.K)
    opts['tail'] = 0.05
    opts['alphas'] = []

    funcs = []
    div_names = []
    for func_spec in args.div_funcs:
        if func_spec.startswith('renyi:'):
            alphas = [float(a) for a in func_spec[len('renyi:'):].split(',')]
            opts['alphas'] = np.sort(alphas)
            func_spec = 'renyi'
            funcs.append(func_mapping['renyi'])
            for alpha in opts['alphas']:
                for K in opts['Ks']:
                    div_names.append('NP-R[a={},K={}]'.format(alpha, K))
        else:
            funcs.append(func_mapping[func_spec])
            name = name_mapping[func_spec]
            div_names.extend(['{}[K={}]'.format(name, K) for K in opts['Ks']])

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
