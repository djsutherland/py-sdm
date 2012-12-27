#!/usr/bin/env python
'''
Code to do classification with support vector machines, as in
    Dougal J. Sutherland, Liang Xiong, Barnabas Poczos, Jeff Schneider.
    Kernels on Sample Sets via Nonparametric Divergence Estimates.
    http://arxiv.org/abs/1202.0302

Code by Dougal J. Sutherland - dsutherl@cs.cmu.edu

Written for Python 2.7 but with Python 3.2+ compatability in mind; currently
untested on 3.x because I only have 3.3 installed, and h5py is still broken
there. Uses some functionality not included in 2.6, but replacing the
itertools.combinations_with_replacement() and installing the argparse package
should probably be enough to get that to work if you need it to.
'''

from __future__ import division, print_function

from functools import partial
import itertools
import sys
import weakref

import h5py
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.base
from sklearn import cross_validation as cv
from sklearn import svm

from get_divs import ForkedData, get_divs, get_pool, progressbar_and_updater, \
                     TAIL_DEFAULT, read_cell_array, \
                     positive_int, positive_float, portion, is_integer_type

DEFAULT_SVM_CACHE = 1000
DEFAULT_SVM_TOL = 1e-3
DEFAULT_C_VALS = tuple(2.0 ** np.arange(-9, 19, 3))
DEFAULT_SIGMA_VALS = tuple(2.0 ** np.arange(-4, 11, 2))
DEFAULT_K = 3
DEFAULT_TUNING_FOLDS = 3

def get_status_fn(val):
    if val is True:
        return partial(print, file=sys.stderr)
    elif val is None:
        return lambda *args, **kwargs: None
    else:
        return val

################################################################################
### PSD projection and friends

def project_psd(mat, min_eig=0, destroy=False):
    '''
    Project a real symmetric matrix to PSD by discarding any negative
    eigenvalues from its spectrum. Passing min_eig > 0 lets you similarly make
    it positive-definite, though this may not technically be a projection...?

    Symmetrizes the matrix before projecting.

    If destroy is True, turns the passed-in matrix into gibberish. If the
    matrix is very large, passing in a weakref.proxy to it will use the least
    amount of memory.
    '''
    if not destroy:
        mat = mat.copy()
    mat += mat.T
    mat /= 2

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat)
    if vals.min() < min_eig:
        del mat
        mat = np.dot(vecs, np.dot(np.diag(np.maximum(vals, min_eig)), vecs.T))
        del vals, vecs
        mat += mat.T
        mat /= 2
    return mat

def make_km(divs, sigma):
    # pass through a Gaussian
    km = divs / sigma # makes a copy
    km **= 2
    km /= -2
    np.exp(km, km) # inplace

    # PSD projection
    return project_psd(weakref.proxy(km), destroy=True)

def split_km(km, train_idx, test_idx):
    train_km = np.ascontiguousarray(km[np.ix_(train_idx, train_idx)])
    test_km = np.ascontiguousarray(km[np.ix_(test_idx, train_idx)])
    return train_km, test_km


################################################################################
### parameter tuning


def try_params(km, labels, train_idx, test_idx, C, params):
    train_km, test_km = split_km(km.value, train_idx, test_idx)

    clf = svm.SVC(C=C, **params)
    clf.fit(train_km, labels.value[train_idx])

    preds = clf.predict(test_km)
    return np.mean(preds == labels.value[test_idx])

def _assign_score(scores, C_vals, sigma_vals, print_fn,
                 C_idx, sigma_idx, f_idx, val):
    scores[C_idx, sigma_idx, f_idx] = val
    print_fn('C {}, sigma {}, fold {}: acc {}'.format(
        C_vals[C_idx], sigma_vals[sigma_idx], f_idx, val))

def tune_params(divs, labels,
                num_folds=DEFAULT_TUNING_FOLDS,
                n_proc=None,
                C_vals=DEFAULT_C_VALS,
                sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                weight_classes=False,
                cache_size=DEFAULT_SVM_CACHE,
                svm_tol=DEFAULT_SVM_TOL,
                status_fn=True,
                progressbar=None):

    if progressbar is None:
        progressbar = status_fn is True
    status_fn = get_status_fn(status_fn)

    C_vals = np.asarray(C_vals)
    sigma_vals = np.asarray(sigma_vals)
    if scale_sigma:
        sigma_vals *= np.median(divs[divs > 0])

    if C_vals.size <= 1 and sigma_vals.size <= 1:
        # no tuning necessary
        return C_vals[0], sigma_vals[0]

    num_bags = divs.shape[0]
    assert divs.ndim == 2 and divs.shape[1] == num_bags
    assert labels.shape == (num_bags,)

    svm_params = dict(
        cache_size=cache_size,
        class_weight='auto' if weight_classes else None,
        kernel='precomputed',
        tol=svm_tol,
    )

    # get kernel matrices for the sigma vals we're trying
    # TODO: could be more careful about making copies here
    sigma_kms = {}
    for sigma in sigma_vals:
        status_fn('Projecting: sigma = {}'.format(sigma))
        sigma_kms[sigma] = ForkedData(make_km(divs, sigma))

    labels_d = ForkedData(labels)

    # try each sigma/C combination and see how they do
    scores = np.empty((C_vals.size, sigma_vals.size, num_folds))
    scores.fill(np.nan)
    assign_score = partial(_assign_score, scores, C_vals, sigma_vals, status_fn)
    if progressbar:
        assign_score_ = assign_score
        pbar, tick_pbar = progressbar_and_updater(
                maxval=len(C_vals) * len(sigma_vals) * num_folds)
        def assign_score(*args, **kwargs):
            assign_score_(*args, **kwargs)
            tick_pbar()

    status_fn('Cross-validating parameter sets...')
    jobs = itertools.product(enumerate(C_vals), enumerate(sigma_vals))
    folds = list(enumerate(cv.KFold(n=num_bags, k=num_folds, shuffle=True)))
    with get_pool(n_proc) as pool:
        for (C_idx, C), (sigma_idx, sigma) in jobs:
            for f_idx, (train, test) in folds:
                set_res = partial(assign_score, C_idx, sigma_idx, f_idx)
                pool.apply_async(try_params,
                    [sigma_kms[sigma], labels_d, train, test, C, svm_params],
                    callback=set_res)

    if progressbar:
        pbar.finish()

    # figure out which ones were best
    # TODO: randomize when there are ties...
    cv_means = scores.mean(axis=-1)
    best_sigma, best_C = np.unravel_index(cv_means.argmax(), cv_means.shape)

    return sigma_vals[best_sigma], C_vals[best_C]


################################################################################
### Main dealio

# TODO: SDM class that does induction

class SupportDistributionMachine(sklearn.base.BaseEstimator):
    def __init__(self,
                 div_func='renyi:.9',
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 C_vals=DEFAULT_C_VALS,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 weight_classes=False,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 status_fn=False, progressbar=None,
                 tail=TAIL_DEFAULT):
        self.div_func = div_func
        self.K = K
        self.tuning_folds = tuning_folds
        self.n_proc = n_proc
        self.C_vals = C_vals
        self.sigma_vals = sigma_vals
        self.scale_sigma = scale_sigma
        self.weight_classes = weight_classes
        self.cache_size = cache_size
        self.tuning_cache_size = tuning_cache_size
        self.svm_tol = svm_tol
        self.tuning_svm_tol = tuning_svm_tol
        self._status_fn = status_fn
        self._progressbar = progressbar
        self.tail = tail

    @property
    def status_fn(self):
        return get_status_fn(self._status_fn)

    @property
    def progressbar(self):
        if self._progressbar is None:
            return self._status_fn is True
        else:
            return self._progressbar

    def fit(self, X, y, divs=None):
        '''
        X: a list of row-instance data matrices, with common dimensionality

        y: a vector of nonnegative integer class labels.
            -1 corresponds to data that should be used semi-supervised, ie
            used in projecting the Gram matrix, but not in training the SVM.
            To do transduction, call fit() with the test data labeled as -1
            and then call predict() on the test data (or use transduct() below).

        divs: precomputed divergences among the passed points
        '''
        n_bags = len(X)

        y = np.squeeze(y)
        assert is_integer_type(y)
        assert y.shape == (n_bags,)
        assert np.all(y >= -1)

        train_idx = y != -1
        train_y = y[train_idx]
        assert train_y.size >= 2

        # get divergences
        if divs is None:
            self.status_fn('Getting divergences...')
            divs = np.squeeze(get_divs(
                    X, specs=[self.div_func], Ks=[self.K],
                    n_proc=self.n_proc, tail=self.tail,
                    status_fn=self.status_fn, progressbar=self.progressbar))
        else:
            self.status_fn('Using passed-in divergences...')
            assert divs.shape == (n_bags, n_bags)

        # tune params
        self.status_fn('Tuning SVM parameters...')
        self.sigma_, self.C_ = tune_params(
                divs=np.ascontiguousarray(divs[np.ix_(train_idx, train_idx)]),
                labels=train_y,
                num_folds=self.tuning_folds,
                n_proc=self.n_proc,
                C_vals=self.C_vals,
                sigma_vals=self.sigma_vals, scale_sigma=self.scale_sigma,
                weight_classes=self.weight_classes,
                cache_size=self.tuning_cache_size,
                svm_tol=self.tuning_svm_tol,
                status_fn=self.status_fn,
                progressbar=self.progressbar)
        self.status_fn('Selected sigma {}, C {}'.format(self.sigma_, self.C_))

        # project the final Gram matrix
        self.status_fn('Doing final projection')
        train_km = np.ascontiguousarray(
                make_km(divs, self.sigma_)[np.ix_(train_idx, train_idx)])

        # train the selected SVM
        self.status_fn('Training final SVM')
        clf = svm.SVC(
                C=self.C_,
                cache_size=self.cache_size,
                class_weight='auto' if self.weight_classes else None,
                tol=self.svm_tol,
                kernel='precomputed',
        )
        clf.fit(train_km, train_y)
        self.svm_ = clf


    def predict(self, data):
        # TODO: find the new divergences
        raise NotImplementedError
        test_km = None
        return self.svm_.predict(test_km)


def transduct(train_bags, train_labels, test_bags,
              div_func='renyi:.9',
              K=DEFAULT_K,
              tuning_folds=DEFAULT_TUNING_FOLDS,
              n_proc=None,
              C_vals=DEFAULT_C_VALS,
              sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
              weight_classes=False,
              cache_size=DEFAULT_SVM_CACHE, tuning_cache_size=DEFAULT_SVM_CACHE,
              svm_tol=DEFAULT_SVM_TOL, tuning_svm_tol=DEFAULT_SVM_TOL,
              status_fn=True,
              progressbar=None,
              tail=TAIL_DEFAULT,
              divs=None,
              return_config=False):
    # TODO: support non-Gaussian kernels
    # TODO: support CVing between multiple div funcs, values of K
    # TODO: support more SVM options

    if progressbar is None:
        progressbar = status_fn is True
    status_fn = get_status_fn(status_fn)

    num_train = len(train_bags)
    train_labels = np.squeeze(train_labels)
    assert train_labels.shape == (num_train,)

    if divs is None:
        status_fn('Getting divergences...')
        divs = np.squeeze(get_divs(
                train_bags + test_bags,
                specs=[div_func], Ks=[K],
                n_proc=n_proc, tail=tail,
                status_fn=status_fn, progressbar=progressbar))
    else:
        status_fn('Using passed-in divergences...')
        n_bags = len(train_bags) + len(test_bags)
        assert divs.shape == (n_bags, n_bags)

    status_fn('Tuning parameters...')
    # TODO: print partial results to somewhere else
    sigma, C = tune_params(
            divs=np.ascontiguousarray(divs[:num_train, :num_train]),
            labels=train_labels,
            num_folds=tuning_folds,
            n_proc=n_proc,
            C_vals=C_vals,
            sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            weight_classes=weight_classes,
            cache_size=tuning_cache_size,
            svm_tol=tuning_svm_tol,
            status_fn=status_fn,
            progressbar=progressbar)
    status_fn('Selected sigma {}, C {}'.format(sigma, C))

    status_fn('Doing final projection')
    train_km, test_km = split_km(
            make_km(divs, sigma),
            xrange(num_train),
            xrange(num_train, divs.shape[0]))

    status_fn('Training final SVM')
    clf = svm.SVC(
            C=C,
            cache_size=cache_size,
            class_weight='auto' if weight_classes else None,
            tol=svm_tol,
            kernel='precomputed',
    )
    clf.fit(train_km, train_labels)

    preds = clf.predict(test_km)
    return (preds, (sigma, C)) if return_config else preds

################################################################################
### Command-line interface

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description='Performs support distribution machine classification.')

    _def = "(default %(default)s)."

    parser.add_argument('input_file',
        help="The input HDF5 file (e.g. a .mat file with -v7.3).")
    parser.add_argument('train_bags_name',
        help="The name of a cell array of row-instance data matrices.")
    parser.add_argument('test_bags_name',
        help="The name of a cell array of row-instance data matrices.")
    parser.add_argument('train_labels_name',
        help="The name of a vector of training labels (integers).")
    parser.add_argument('output_file', nargs='?',
        help="Name of the output file; defaults to input_file.py_divs.mat.")

    parser.add_argument('--n-proc', type=positive_int, default=None,
        help="Number of processes to use; default is as many as CPU cores.")
    parser.add_argument('--n-points', type=positive_int, default=None,
        help="The number of points to use per group; defaults to all.")


    parser.add_argument('--div-func', '-d', default='renyi:.9',
        help="The divergence function to use; default %(default)s.")

    parser.add_argument('-K', type=positive_int, default=DEFAULT_K,
        help="How many nearest neighbors to use; default %(default)s.")

    parser.add_argument('--svm-tol', type=positive_float, default=DEFAULT_SVM_TOL,
        help="SVM solution tolerance " + _def)
    parser.add_argument('--cache-size', type=positive_float, default=DEFAULT_SVM_CACHE,
        help="Size of the SVM cache, in megabytes " + _def)

    parser.add_argument('--tuning-folds', '-F', type=positive_int,
        default=DEFAULT_TUNING_FOLDS,
        help="Number of CV folds to use in evaluating parameters " + _def)
    parser.add_argument('--tuning-svm-tol',
        type=positive_float, default=DEFAULT_SVM_TOL,
        help="SVM solution tolerance in tuning " + _def)
    parser.add_argument('--tuning-cache-size', type=positive_float,
        default=DEFAULT_SVM_CACHE,
        help="Size of tuning SVMs' cache, in megabytes " + _def)

    s = parser.add_mutually_exclusive_group()
    s.add_argument('--weight-classes', action='store_true', default=False)
    s.add_argument('--no-weight-classes', action='store_false',
        dest='weight_classes')

    parser.add_argument('--c-vals', '-C', type=positive_float, nargs='+',
        default=DEFAULT_C_VALS, metavar='C')
    parser.add_argument('--sigma-vals', '-S', type=positive_float, nargs='+',
        default=DEFAULT_SIGMA_VALS, metavar='SIGMA')

    s = parser.add_mutually_exclusive_group()
    s.add_argument('--scale-sigma', action='store_true', default=True)
    s.add_argument('--no-scale-sigma', action='store_false', dest='scale_sigma')

    parser.add_argument('--trim-tails', type=portion, metavar='PORTION',
        default=TAIL_DEFAULT,
        help="How much to trim off ends of things we take the mean of " + _def)

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file + '.sdm_results.mat'

    args.c_vals = np.sort(args.c_vals)
    args.sigma_vals = np.sort(args.sigma_vals)

    return args


def main():
    args = parse_args()

    status_fn = get_status_fn(True)

    # TODO: use logging module to save partial results

    status_fn('Reading inputs...')
    with h5py.File(args.input_file, 'r') as f:
        train_bags = read_cell_array(f, f[args.train_bags_name], args.n_points)
        train_labels = f[args.train_labels_name].value
        test_bags = read_cell_array(f, f[args.test_bags_name], args.n_points)

    # TODO: optionally cache divergences

    preds, (sigma, C) = transduct(
            train_bags, train_labels, test_bags,
            div_func=args.div_func,
            K=args.K,
            tuning_folds=args.tuning_folds,
            n_proc=args.n_proc,
            C_vals=args.c_vals,
            sigma_vals=args.sigma_vals, scale_sigma=args.scale_sigma,
            weight_classes=args.weight_classes,
            cache_size=args.cache_size,
            tail=args.trim_tails,
            return_config=True)

    out = {
        'div_func': args.div_func,
        'K': args.K,
        'C_vals': args.c_vals,
        'sigma_vals': args.sigma_vals,
        'scale_sigma': args.scale_sigma,
        'C': C,
        'sigma': sigma,
        'preds': preds,
    }
    status_fn('Saving output to {}'.format(args.output_file))
    scipy.io.savemat(args.output_file, out, oned_as='column')

if __name__ == '__main__':
    main()
