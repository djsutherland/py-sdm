#!/usr/bin/env python
'''
Code to do classification with divergence kernels in SVMs, as in
    Dougal J. Sutherland, Liang Xiong, Barnabas Poczos, Jeff Schneider.
    Kernels on Sample Sets via Nonparametric Divergence Estimates.
    http://arxiv.org/abs/1202.0302
'''

from __future__ import division, print_function

from collections import Counter
from functools import partial
import itertools
from operator import itemgetter
import os
import random
import sys
import warnings
import weakref

import h5py
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.base
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ConvergenceWarning
from sklearn import svm  # NOTE: needs version 0.13+ for svm iter limits

from .utils import positive_int, positive_float, portion, is_integer_type
from .mp_utils import ForkedData, get_pool, progressbar_and_updater
from .np_divs import (estimate_divs,
                      FIX_MODE_DEFAULT, FIX_TERM_MODES, TAIL_DEFAULT,
                      read_cell_array, subset_data,
                      check_h5_settings, add_to_h5_cache, normalize_div_name)

# TODO: better logging
# TODO: support getting decision values / probabilities

DEFAULT_SVM_CACHE = 1000
DEFAULT_SVM_TOL = 1e-3
DEFAULT_SVM_ITER = 10 ** 6
DEFAULT_SVM_ITER_TUNING = 1000
DEFAULT_SVM_SHRINKING = True

DEFAULT_C_VALS = tuple(2.0 ** np.arange(-9, 19, 3))
DEFAULT_SIGMA_VALS = tuple(2.0 ** np.arange(-4, 11, 2))
DEFAULT_SVR_NU_VALS = (0.2, 0.3, 0.5, 0.7)
DEFAULT_K = 3
DEFAULT_TUNING_FOLDS = 3

def get_status_fn(val):
    if val is True:
        return partial(print, file=sys.stderr)
    elif val is None:
        return lambda *args, **kwargs: None
    else:
        return val

def is_categorical_type(ary):
    return is_integer_type(ary) or np.asanyarray(ary).dtype.kind == 'b'


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


def make_km(divs, sigma, project=True, destroy=False):
    # pass through a Gaussian
    km = divs if destroy else divs.copy()
    km /= sigma
    km **= 2
    km /= -2
    np.exp(km, km)  # inplace

    # PSD projection
    if project:
        # FIXME: weakref.proxy doesn't actually do anything here since the
        #        caller still has a reference. can we work around that?
        km = project_psd(weakref.proxy(km), destroy=True)

    return km


def split_km(km, train_idx, test_idx):
    train_km = np.ascontiguousarray(km[np.ix_(train_idx, train_idx)])
    test_km = np.ascontiguousarray(km[np.ix_(test_idx, train_idx)])
    return train_km, test_km


################################################################################
### Cached divs helper

def get_divs_cache(bags, div_func, K, cache_filename=None,
                   names=None, cats=None,
                   fix_mode=FIX_MODE_DEFAULT, tail=TAIL_DEFAULT, min_dist=None,
                   n_proc=None, status_fn=True, progressbar=None):
    # TODO: support loading subsets of the file, or reordered, based on names

    status = get_status_fn(status_fn)

    if cache_filename and os.path.exists(cache_filename):
        path = '{}/{}'.format(div_func, K)
        with h5py.File(cache_filename, 'r') as f:
            check_h5_settings(f, n=len(bags), dim=bags[0].shape[1],
                fix_mode=fix_mode, tail=tail, min_dist=min_dist,
                names=names, cats=cats)
            if path in f:
                divs = f[path]
                # assert divs.shape == (len(bags), len(bags)) # in check

                status("Loading divs from cache '{}'".format(cache_filename))
                return divs[...]

    divs = np.squeeze(estimate_divs(
            bags, specs=[div_func], Ks=[K],
            n_proc=n_proc, fix_mode=fix_mode, tail=tail, min_dist=min_dist,
            status_fn=status_fn, progressbar=progressbar))

    if cache_filename:
        status("Saving divs to cache '{}'".format(cache_filename))
        with h5py.File(cache_filename) as f:
            add_to_h5_cache(f, {(div_func, K): divs},
                            dim=bags[0].shape[1],
                            fix_mode=fix_mode, tail=tail, min_dist=min_dist,
                            names=names, cats=cats)

    return divs


################################################################################
### Parameter tuning

# TODO: make more general by using score(), generalize params

def try_params_SVC(km, train_idx, test_idx, C, labels, params):
    '''Try params in an SVC; returns classification accuracy.'''
    train_km, test_km = split_km(km.value, train_idx, test_idx)

    clf = svm.SVC(C=C, **params)
    clf.fit(train_km, labels.value[train_idx])

    preds = clf.predict(test_km)
    assert not np.any(np.isnan(preds))
    score = np.mean(preds == labels.value[test_idx])
    return score, clf.fit_status_

def try_params_NuSVR(km, train_idx, test_idx, C, nu, labels, params):
    '''Try params in a NuSVR; returns negative of mean squared error.'''
    train_km, test_km = split_km(km.value, train_idx, test_idx)

    clf = svm.NuSVR(C=C, nu=nu, **params)
    clf.fit(train_km, labels.value[train_idx])

    preds = clf.predict(test_km)
    assert not np.any(np.isnan(preds))
    score = -np.mean((preds - labels.value[test_idx]) ** 2)
    return score, clf.fit_status_

# generalize for the params we're going over (use sklearn helpers?)
def tune_params(divs, labels,
                mode='SVC',
                num_folds=DEFAULT_TUNING_FOLDS,
                n_proc=None,
                C_vals=DEFAULT_C_VALS,
                sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                svr_nu_vals=DEFAULT_SVR_NU_VALS,
                weight_classes=False,
                cache_size=DEFAULT_SVM_CACHE,
                svm_tol=DEFAULT_SVM_TOL,
                svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                svm_shrinking=DEFAULT_SVM_SHRINKING,
                status_fn=True,
                progressbar=None):
    if progressbar is None:
        progressbar = status_fn is True
    status_fn = get_status_fn(status_fn)

    assert mode in ('SVC', 'NuSVR')

    C_vals = np.asarray(C_vals)
    sigma_vals = np.asarray(sigma_vals)
    if scale_sigma:
        sigma_vals = sigma_vals * np.median(divs[divs > 0])
        # make sure not to modify the passed-in sigma values...

    if mode == 'NuSVR':
        svr_nu_vals = np.asarray(svr_nu_vals)

    if C_vals.size <= 1 and sigma_vals.size <= 1:
        # no tuning necessary, unless we're regressing and have nu_vals
        if mode == 'SVC':
            return C_vals[0], sigma_vals[0]
        elif mode == 'NuSVR' and svr_nu_vals.size <= 1:
            return C_vals[0], sigma_vals[0], svr_nu_vals[0]

    num_bags = divs.shape[0]
    assert divs.ndim == 2 and divs.shape[1] == num_bags
    assert labels.shape == (num_bags,)

    svm_params = {
        'cache_size': cache_size,
        'kernel': 'precomputed',
        'tol': svm_tol,
        'max_iter': svm_max_iter,
        'shrinking': svm_shrinking,
        # 'verbose': True,
    }
    if mode == 'SVC':
        svm_params['class_weight'] = 'auto' if weight_classes else None

    # get kernel matrices for the sigma vals we're trying
    # TODO: could be more careful about making copies here
    sigma_kms = {}
    status_fn('Projecting...')
    for sigma in sigma_vals:
        #status_fn('Projecting: sigma = {}'.format(sigma))
        sigma_kms[sigma] = ForkedData(make_km(divs, sigma))

    labels_d = ForkedData(labels)

    # try each param combination and see how they do
    if mode == 'NuSVR':
        shape = (sigma_vals.size, C_vals.size, svr_nu_vals.size, num_folds)
    else:
        shape = (sigma_vals.size, C_vals.size, num_folds)
    scores = np.empty(shape)
    scores.fill(np.nan)

    status_fn('Cross-validating parameter sets...')

    conv_warning_counter = itertools.count()
    if progressbar:
        pbar, tick_pbar = progressbar_and_updater(maxval=scores.size)

    def assign_score(indices, val_and_status):
        val, status = val_and_status
        scores[indices] = val
        if status:
            next(conv_warning_counter)
        if progressbar:
            tick_pbar()

    if mode == 'NuSVR':
        jobs = itertools.product(
            enumerate(sigma_vals), enumerate(C_vals), enumerate(svr_nu_vals))
    else:
        jobs = itertools.product(enumerate(sigma_vals), enumerate(C_vals))
    folds = list(enumerate(KFold(n=num_bags, n_folds=num_folds, shuffle=True)))

    try_params = partial(try_params_SVC if mode == 'SVC' else try_params_NuSVR,
                  labels=labels_d, params=svm_params)

    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    ignore_conv = warnings.filters[0]

    with get_pool(n_proc) as pool:
        for job in jobs:
            indices, param_vals = zip(*job)
            sigma = param_vals[0]

            for f_idx, (train, test) in folds:
                set_res = partial(assign_score, (indices + (f_idx,)))
                args = (sigma_kms[sigma], train, test) + param_vals[1:]
                pool.apply_async(try_params, args, callback=set_res)

    if progressbar:
        pbar.finish()

    warnings.filters.remove(ignore_conv)
    status_fn('{} SVMs terminated early, after {:,} steps'.format(
        next(conv_warning_counter), svm_max_iter))

    # figure out which ones were best
    assert not np.any(np.isnan(scores))
    cv_means = scores.mean(axis=-1)
    top_elts = cv_means == cv_means.max()
    best_indices = random.choice(np.transpose(top_elts.nonzero()))
    if mode == 'NuSVR':
        best_sigma, best_C, best_svr_nu = best_indices
        return sigma_vals[best_sigma], C_vals[best_C], svr_nu_vals[best_svr_nu]
    else:
        best_sigma, best_C = best_indices
        return sigma_vals[best_sigma], C_vals[best_C]


################################################################################
### Main dealio

class SupportDistributionMachine(sklearn.base.BaseEstimator):
    # TODO: split into subclasses for classification, regression, one-class
    # TODO: support non-Gaussian kernels
    # TODO: support CVing between multiple div funcs, values of K
    # TODO: support more SVM options
    def __init__(self,
                 div_func='renyi:.9',
                 K=DEFAULT_K,
                 mode='SVC',
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 C_vals=DEFAULT_C_VALS,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 svr_nu_vals=DEFAULT_SVR_NU_VALS,
                 weight_classes=False,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 status_fn=None, progressbar=None,
                 fix_mode=FIX_MODE_DEFAULT, tail=TAIL_DEFAULT, min_dist=None,
                 save_bags=True):
        assert mode in ('SVC', 'NuSVR')
        self.mode = mode
        self.div_func = div_func
        self.K = K
        self.tuning_folds = tuning_folds
        self.n_proc = n_proc
        self.C_vals = C_vals
        self.sigma_vals = sigma_vals
        self.scale_sigma = scale_sigma
        self.svr_nu_vals = svr_nu_vals
        self.weight_classes = weight_classes
        self.cache_size = cache_size
        self.tuning_cache_size = tuning_cache_size
        self.svm_tol = svm_tol
        self.tuning_svm_tol = tuning_svm_tol
        self.svm_max_iter = svm_max_iter
        self.tuning_svm_max_iter = tuning_svm_max_iter
        self.svm_shrinking = svm_shrinking
        self._status_fn = status_fn
        self._progressbar = progressbar
        self.fix_mode = FIX_MODE_DEFAULT
        self.tail = tail
        self.min_dist = min_dist
        self.save_bags = save_bags

    @property
    def classifier(self):
        return self.mode == 'SVC'

    @property
    def regressor(self):
        return self.mode == 'NuSVR'

    @property
    def status_fn(self):
        return get_status_fn(self._status_fn)

    @status_fn.setter
    def status_fn(self, value):
        self._status_fn = value

    @property
    def progressbar(self):
        if self._progressbar is None:
            return self._status_fn is True
        else:
            return self._progressbar

    @progressbar.setter
    def progressbar(self, value):
        self._progressbar = value

    def fit(self, X, y, divs=None, divs_cache=None, names=None, cats=None,
            ret_km=False):
        '''
        X: a list of row-instance data matrices, with common dimensionality

        If classifying, y should be a vector of nonnegative integer class labels
            -1 corresponds to data that should be used semi-supervised, ie
            used in projecting the Gram matrix, but not in training the SVM.
        If regressing, y should be a vector of real-valued class labels
            nan is the corresponding semi-supervised indicator value

        divs: precomputed divergences among the passed points

        divs_cache: a filename for a cache file. Note that this needs to be
                    on the TRAINING data only, in the same order.
        names, cats: optional metadata to verify the cache file is actually
                     for the right data (highly recommended if available)
        '''
        n_bags = len(X)

        y = np.squeeze(y)
        assert y.shape == (n_bags,)
        if self.classifier:
            assert is_categorical_type(y)
            assert np.all(y >= -1)

            train_idx = y != -1
        else:
            train_idx = ~np.isnan(y)

        train_y = y[train_idx]
        assert train_y.size >= 2
        assert not np.all(train_y == train_y[0])
        if self.save_bags:
            self.train_bags_ = itemgetter(*train_idx.nonzero()[0])(X)

        # get divergences
        if divs is None:
            self.status_fn('Getting divergences...')
            divs = get_divs_cache(X, div_func=self.div_func, K=self.K,
                    cache_filename=divs_cache,
                    n_proc=self.n_proc, min_dist=self.min_dist,
                    fix_mode=self.fix_mode, tail=self.tail,
                    names=names, cats=cats,
                    status_fn=self.status_fn, progressbar=self.progressbar)
        else:
            #self.status_fn('Using passed-in divergences...')
            assert divs.shape == (n_bags, n_bags)

        # tune params
        self.status_fn('Tuning SVM parameters...')
        tuned_params = tune_params(
                divs=np.ascontiguousarray(divs[np.ix_(train_idx, train_idx)]),
                mode=self.mode,
                labels=train_y,
                num_folds=self.tuning_folds,
                n_proc=self.n_proc,
                C_vals=self.C_vals,
                sigma_vals=self.sigma_vals, scale_sigma=self.scale_sigma,
                svr_nu_vals=self.svr_nu_vals,
                weight_classes=self.weight_classes,
                cache_size=self.tuning_cache_size,
                svm_tol=self.tuning_svm_tol,
                svm_max_iter=self.tuning_svm_max_iter,
                svm_shrinking=self.svm_shrinking,
                status_fn=self.status_fn,
                progressbar=self.progressbar)
        if self.mode == 'SVC':
            self.sigma_, self.C_ = tuned_params
            self.status_fn('Chose sigma {}, C {}'.format(*tuned_params))
        elif self.mode == 'NuSVR':
            self.sigma_, self.C_, self.svr_nu_ = tuned_params
            self.status_fn('Chose sigma {}, C {}, nu {}'.format(*tuned_params))
        else:
            raise ValueError

        # project the final Gram matrix
        self.status_fn('Doing final projection')
        full_km = make_km(divs, self.sigma_)
        train_km = np.ascontiguousarray(full_km[np.ix_(train_idx, train_idx)])

        # train the selected SVM
        self.status_fn('Training final SVM')
        params = {
            'cache_size': self.cache_size,
            'tol': self.svm_tol,
            'kernel': 'precomputed',
            'max_iter': self.svm_max_iter,
            'shrinking': self.svm_shrinking,
        }
        if self.mode == 'SVC':
            params['class_weight'] = 'auto' if self.weight_classes else None
            clf = svm.SVC(C=self.C_, **params)
        elif self.mode == 'NuSVR':
            clf = svm.NuSVR(C=self.C_, nu=self.svr_nu_, **params)
        else:
            raise ValueError
        clf.fit(train_km, train_y)
        self.svm_ = clf

        if ret_km:
            return full_km

    def predict(self, data, divs=None, km=None):
        if getattr(self, 'svm_', None) is None:
            raise ValueError("SDM: need to fit before you can predict!")

        if km is not None:
            pass
        elif divs is not None:
            km = make_km(divs, self.sigma_, project=False, destroy=False)
        else:
            if not self.save_bags:
                raise ValueError("SDM that doesn't save_bags can't predict "
                                 "without explicit divs")

            n_train = len(self.train_bags_)
            n_test = len(data)

            self.status_fn('Getting test bag divergences...')

            mask = np.zeros((n_train + n_test, n_train + n_test), dtype=bool)
            mask[:n_train, -n_test:] = True
            mask[-n_test:, :n_train] = True

            divs = np.squeeze(estimate_divs(
                    self.train_bags_ + tuple(data), mask=mask,
                    specs=[self.div_func], Ks=[self.K],
                    n_proc=self.n_proc, min_dist=self.min_dist,
                    fix_mode=self.fix_mode, tail=self.tail,
                    status_fn=self.status_fn, progressbar=self.progressbar))
            divs = (divs[-n_test:, :n_train] + divs[:n_train, -n_test].T) / 2
            km = make_km(divs, self.sigma_, project=False, destroy=True)

        # TODO: smarter projection options for inductive use

        preds = self.svm_.predict(km)
        if self.classifier:
            assert np.all(preds == np.round(preds))
            return preds.astype(int)
        else:
            return preds

    def transduct(self, train_bags, train_labels, test_bags, divs=None,
                  save_fit=False):
        '''
        Trains an SDM transductively, where the kernel matrix is constructed on
        the training + test points, the SVM is trained on training points, and
        predictions are done on the test points.

        The SVM itself is inductive (given the kernel).

        If divs is passed, it should be a div matrix for train_bags + test_bags.
        Transparent caching is not yet supported here because of the re-ordering
        issue.

        By default, the object does not save the fit state and is reset to
        an un-fit state as if it had just been constructed.
        Passing save_fit=True makes the fit persistent.
        '''
        # TODO: support transparent divs caching by passing in indices

        n_train = len(train_bags)
        n_test = len(test_bags)

        train_labels = np.squeeze(train_labels)
        assert train_labels.shape == (n_train,)
        if self.classifier:
            assert is_categorical_type(train_labels)
            assert np.all(train_labels >= 0)
        else:
            assert np.all(np.isfinite(train_labels))

        combo_bags = train_bags + test_bags

        if not save_fit:
            old_save_bags = self.save_bags
            self.save_bags = False  # avoid keeping copies around

        # make fake labels for test data, so fit() knows what they are
        test_fake_labels = np.empty(n_test, dtype=train_labels.dtype)
        test_fake_labels.fill(-1 if self.classifier else np.nan)
        combo_labels = np.hstack((train_labels, test_fake_labels))

        full_km = self.fit(combo_bags, combo_labels, divs=divs, ret_km=True)
        preds = self.predict(test_bags, km=full_km[-n_test:, :n_train])

        if not save_fit:
            self.save_bags = old_save_bags
            for attr_name in dir(self):
                if attr_name.endswith('_') and not attr_name.startswith('_'):
                    delattr(self, attr_name)
        return preds


################################################################################
### Cross-validation helper

def crossvalidate(bags, labels,
        num_folds=10, stratified_cv=False,
        mode='SVC',
        div_func='renyi:.9',
        K=DEFAULT_K,
        tuning_folds=DEFAULT_TUNING_FOLDS,
        project_all=True,
        n_proc=None,
        C_vals=DEFAULT_C_VALS,
        sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
        svr_nu_vals=DEFAULT_SVR_NU_VALS,
        weight_classes=False,
        cache_size=DEFAULT_SVM_CACHE, tuning_cache_size=DEFAULT_SVM_CACHE,
        svm_tol=DEFAULT_SVM_TOL, tuning_svm_tol=DEFAULT_SVM_TOL,
        svm_max_iter=DEFAULT_SVM_ITER,
        tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
        svm_shrinking=DEFAULT_SVM_SHRINKING,
        status_fn=True,
        progressbar=None,
        fix_mode=FIX_MODE_DEFAULT, tail=TAIL_DEFAULT, min_dist=None,
        divs=None,
        divs_cache=None, names=None, cats=None):

    # TODO: allow specifying what the folds should be
    # TODO: optionally return params for each fold, what the folds were, ...

    args = locals()
    opts = dict((v, args[v]) for v in
        ['mode', 'div_func', 'K', 'tuning_folds', 'n_proc',
         'C_vals', 'sigma_vals', 'scale_sigma', 'svr_nu_vals', 'weight_classes',
         'cache_size', 'tuning_cache_size', 'svm_tol', 'tuning_svm_tol',
         'svm_max_iter', 'tuning_svm_max_iter', 'svm_shrinking',
         'status_fn', 'progressbar',
         'fix_mode', 'tail', 'min_dist'])

    status = get_status_fn(status_fn)

    num_bags = len(bags)
    dim = bags[0].shape[1]
    assert all(bag.ndim == 2 and bag.shape[1] == dim and bag.shape[0] > 0
               for bag in bags)

    classifier = mode == 'SVC'

    labels = np.squeeze(labels)
    assert labels.shape == (num_bags,)
    if classifier:
        assert is_categorical_type(labels)
        assert np.all(labels >= 0)
    else:
        assert np.all(np.isfinite(labels))

    if divs is None:
        status('Getting divergences...')
        divs = get_divs_cache(bags, div_func=div_func, K=K,
                cache_filename=divs_cache, n_proc=n_proc,
                fix_mode=fix_mode, tail=tail, min_dist=min_dist,
                names=names, cats=cats,
                status_fn=status_fn, progressbar=progressbar)
    else:
        #status_fn('Using passed-in divergences...')
        assert divs.shape == (num_bags, num_bags)

    if classifier:
        preds = -np.ones(num_bags, dtype=int)
    else:
        preds = np.empty(num_bags)
        preds.fill(np.nan)

    if stratified_cv:
        cv_folds = StratifiedKFold(labels, n_folds=num_folds)
    else:
        cv_folds = KFold(n=num_bags, n_folds=num_folds, shuffle=True)

    for i, (train, test) in enumerate(cv_folds, 1):
        status('')
        status('Starting fold {} / {}'.format(i, num_folds))

        both = np.hstack((train, test))
        train_bags = itemgetter(*train)(bags)
        test_bags = itemgetter(*test)(bags)

        if classifier:
            status('Test distribution: {}'.format(dict(Counter(labels[test]))))

        clf = SupportDistributionMachine(**opts)
        if project_all:
            preds[test] = clf.transduct(train_bags, labels[train], test_bags,
                                        divs=divs[np.ix_(both, both)])
        else:
            clf.fit(train_bags, labels[train], divs=divs[np.ix_(train, train)])
            preds[test] = clf.predict(test_bags)

        if classifier:
            acc = np.mean(preds[test] == labels[test])
            status('Fold accuracy: {:.1%}'.format(acc))
        else:
            rmse = np.sqrt(np.mean((preds[test] - labels[test]) ** 2))
            status('Fold RMSE: {}'.format(rmse))

    if classifier:
        return np.mean(preds == labels), preds
    else:
        return np.sqrt(np.mean((preds - labels) ** 2)), preds


################################################################################
### Command-line interface

def parse_args():
    import argparse

    # helper for boolean flags
    # based on http://stackoverflow.com/a/9236426/344821
    class ActionNoYes(argparse.Action):
        def __init__(self, opt_name, off_name=None, dest=None,
                     default=True, required=False, help=None):

            if off_name is None:
                off_name = 'no-' + opt_name
            self.off_name = '--' + off_name

            if dest is None:
                dest = opt_name.replace('-', '_')

            super(ActionNoYes, self).__init__(
                    ['--' + opt_name, '--' + off_name],
                    dest, nargs=0, const=None,
                    default=default, required=required, help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, option_string != self.off_name)

    # component of a help string that adds the default value
    _def = "(default %(default)r)."

    # add common options to a parser
    # would use parents=[...], except for http://bugs.python.org/issue16807
    def add_opts(parser):
        algo = parser.add_argument_group('algorithm options')

        m = algo.add_mutually_exclusive_group()
        m.set_defaults(mode='transduct')
        m.add_argument('--transduct',
            action='store_const', dest='mode', const='transduct',
            help="Operate transductively (project full Gram matrix; default).")
        m.add_argument('--induct',
            action='store_const', dest='mode', const='induct',
                help="Operate inductively (only project training Gram matrix).")

        m = algo.add_mutually_exclusive_group()
        m.set_defaults(svm_mode='SVC')
        m.add_argument('--svc',
            action='store_const', dest='svm_mode', const='SVC',
            help="Use the standard support vector classifier (default).")
        m.add_argument('--nu-svr',
            action='store_const', dest='svm_mode', const='NuSVR',
            help="Use a NuSVR support vector regressor.")

        algo.add_argument('--div-func', '-d', default='renyi:.9',
            type=normalize_div_name,
            help="The divergence function to use " + _def)

        algo.add_argument('-K', type=positive_int, default=DEFAULT_K,
            help="How many nearest neighbors to use " + _def)

        algo.add_argument('--n-points', type=positive_int, default=None,
            help="The number of points to use per group; defaults to all.")

        algo.add_argument('--tuning-folds', '-F', type=positive_int,
            default=DEFAULT_TUNING_FOLDS,
            help="Number of CV folds to use in evaluating parameters " + _def)

        comp = parser.add_argument_group('computation options')
        comp.add_argument('--n-proc', type=positive_int, default=None,
            help="Number of processes to use; default is as many as CPU cores.")

        comp.add_argument('--svm-tol',
            type=positive_float, default=DEFAULT_SVM_TOL,
            help="SVM solution tolerance " + _def)
        g = comp.add_mutually_exclusive_group()
        g.add_argument('--svm-max-iter',
            type=positive_int, default=DEFAULT_SVM_ITER,
            help="Limit on the number of SVM iterations " + _def)
        g.add_argument('--svm-unlimited-iter',
            action='store_const', const=-1, dest='svm_max_iter',
            help="Let the SVM try to iterate until full convergence.")
        comp.add_argument('--cache-size',
            type=positive_float, default=DEFAULT_SVM_CACHE,
            help="Size of the SVM cache, in megabytes " + _def)
        comp._add_action(ActionNoYes('svm-shrinking', default=True,
            help="Use the shrinking heuristics in the SVM (default: do)."))

        comp.add_argument('--tuning-svm-tol',
            type=positive_float, default=DEFAULT_SVM_TOL,
            help="SVM solution tolerance in tuning " + _def)
        comp.add_argument('--tuning-svm-max-iter',
            type=positive_int, default=DEFAULT_SVM_ITER_TUNING,
            help="Limit on the number of SVM iterations in tuning " + _def)
        comp.add_argument('--tuning-svm-unlimited-iter',
            action='store_const', const=-1, dest='tuning_svm_max_iter',
            help="Let the SVM try to iterate until full convergence in tuning.")
        comp.add_argument('--tuning-cache-size', type=positive_float,
            default=DEFAULT_SVM_CACHE,
            help="Size of tuning SVMs' cache, in megabytes " + _def)

        algo._add_action(ActionNoYes('weight-classes', default=False,
            help="Reweight SVM loss to equalize classes (default: don't). "
                 "Only applies to classification."))

        algo.add_argument('--c-vals', '-C', type=positive_float, nargs='+',
            default=DEFAULT_C_VALS, metavar='C',
            help="Values to try for tuning SVM regularization strength " + _def)
        algo.add_argument('--sigma-vals', '-S', type=positive_float, nargs='+',
            default=DEFAULT_SIGMA_VALS, metavar='SIGMA',
            help="Values to try for tuning kernel bandwidth sigma " + _def)
        algo._add_action(ActionNoYes('scale-sigma', default=True,
            help="Scale --sigma-vals by the median nonzero divergence; "
                 "does by default."))
        algo.add_argument('--svr-nu-vals', type=portion, nargs='+',
            default=DEFAULT_SVR_NU_VALS, metavar='NU',
            help="Values to try for tuning the nu of NuSVR, a lower bound on "
                 "the fraction of support vectors " + _def)

        algo.add_argument('--trim-tails', type=portion, metavar='PORTION',
            default=TAIL_DEFAULT,
            help="How much to trim when using a trimmed mean estimator " + _def)
        algo.add_argument('--trim-mode',
            choices=FIX_TERM_MODES, default=FIX_MODE_DEFAULT,
            help="Whether to trim or clip ends; default %(default)s.")
        algo.add_argument('--min-dist', type=float, default=None,
            help="Protect against identical points by making sure kNN "
                 "distances are always at least this big. Default: the smaller "
                 "of .01 and 10 ^ (100 / dim).")

    ### the top-level parser
    parser = argparse.ArgumentParser(
            description='Performs support distribution machine classification.')
    subparsers = parser.add_subparsers(dest='subcommand',
            help="The kind of action to perform.")

    ### parser for the prediction task
    parser_pred = subparsers.add_parser('predict',
            help="Train on labeled training data, predict on test data.")
    parser_pred.set_defaults(func=do_predict)

    io = parser_pred.add_argument_group('input/output options')
    io.add_argument('input_file',
        help="The input HDF5 file (e.g. a .mat file with -v7.3). Prediction "
             'currently only supports the "matlab-style" format.')

    io.add_argument('--train-bags-name', default='train_bags',
        help="The name of a cell array of row-instance data matrices " + _def)
    io.add_argument('--test-bags-name', default='test_bags',
        help="The name of a cell array of row-instance data matrices " + _def)
    io.add_argument('--train-labels-name', default='train_labels',
        help="The name of a vector of training labels, int if classifying, "
             "float if regressing " + _def)

    io.add_argument('--output-file', required=False,
        help="Name of the output file; defaults to input_file.sdm_preds.mat.")

    io.add_argument('--div-cache-file',
        help="An HDF5 file that serves as a cache of divergences.")

    add_opts(parser_pred)

    ### parser for the cross-validation task
    parser_cv = subparsers.add_parser('cv',
            help="Cross-validate predictions on fully labeled data.")
    parser_cv.set_defaults(func=do_cv)

    io = parser_cv.add_argument_group('input/output options')
    io.add_argument('input_file',
        help="The input HDF5 file (e.g. a .mat file with -v7.3).")
    io.add_argument('--input-format',
        choices=['matlab', 'python'], default='python',
        help="Whether the features file was generated by the matlab code or "
             "the python code; default python.")

    io.add_argument('--bags-name', default='features',
        help="The name of a cell array of row-instance data matrices " + _def
             + " Only used for matlab format.")
    io.add_argument('--labels-name', default=None,
        help="The name of a vector of training labels "
             "(integers if classifying, reals if regressing). "
             "If matlab format, default 'cats'; "
             "if classifying in python format, default is --classify-by-cats; "
             "if regressing in python format, no default.")
    io.add_argument('--classify-by-cats',
        dest='labels_name', action='store_const', const=None,
        help="When classifying in python format, use as labels the category "
             "names (the default).")

    io.add_argument('--output-file', required=False,
        help="Name of the output file; defaults to input_file.sdm_cv.mat.")

    io.add_argument('--div-cache-file',
        help="An HDF5 file that serves as a cache of divergences.")

    cv = parser_cv.add_argument_group('cross-validation options')
    cv.add_argument('--cv-folds', '-f', type=positive_int, default=10,
        help="The number of cross-validation folds " + _def)
    cv._add_action(ActionNoYes('stratified-cv', default=False,
        help="Choose CV folds to keep approximately the same class "
             "distribution in each one (by default, doesn't)."))

    add_opts(parser_cv)

    ### parse the arguments and do some post-processing
    args = parser.parse_args()

    if args.output_file is None:
        suffixes = {
            'predict': '.sdm_preds.mat',
            'cv': '.sdm_cv.mat',
        }
        args.output_file = args.input_file + suffixes[args.subcommand]

    args.c_vals = np.sort(args.c_vals)
    args.sigma_vals = np.sort(args.sigma_vals)

    return args


def opts_dict(args):
    return dict(
        mode=args.svm_mode,
        div_func=args.div_func,
        K=args.K,
        tuning_folds=args.tuning_folds,
        n_proc=args.n_proc,
        C_vals=args.c_vals,
        sigma_vals=args.sigma_vals, scale_sigma=args.scale_sigma,
        svr_nu_vals=args.svr_nu_vals,
        weight_classes=args.weight_classes,
        cache_size=args.cache_size,
        tuning_cache_size=args.tuning_cache_size,
        svm_tol=args.svm_tol,
        tuning_svm_tol=args.tuning_svm_tol,
        svm_max_iter=args.svm_max_iter,
        tuning_svm_max_iter=args.tuning_svm_max_iter,
        svm_shrinking=args.svm_shrinking,
        tail=args.trim_tails,
        fix_mode=args.trim_mode,
        min_dist=args.min_dist,
    )


def do_predict(args):
    status_fn = get_status_fn(True)

    status_fn('Reading inputs...')
    with h5py.File(args.input_file, 'r') as f:
        train_bags = read_cell_array(f, f[args.train_bags_name])
        train_labels = f[args.train_labels_name][...]
        test_bags = read_cell_array(f, f[args.test_bags_name])
        if args.n_points:
            train_bags = subset_data(train_bags, args.n_points)
            test_bags = subset_data(test_bags, args.n_points)

    assert np.all(train_labels == np.round(train_labels))
    train_labels = train_labels.astype(int)

    clf = SupportDistributionMachine(status_fn=True, **opts_dict(args))
    if args.mode == 'transduct':
        if args.div_cache_file:
            msg = ("Can't currently use divergence cache when transducting and "
                   "not cross-validating...out of laziness, so if you want "
                   "this, just complain.")
            warnings.warn(msg)
        # TODO: support partial caching of divs here
        preds = clf.transduct(train_bags, train_labels, test_bags)
    elif args.mode == 'induct':
        clf.fit(train_bags, train_labels,
                divs_cache=args.div_cache_file, cats=train_labels)
        preds = clf.predict(test_bags)
    sigma = clf.sigma_
    C = clf.C_

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


def do_cv(args):
    status_fn = get_status_fn(True)

    status_fn('Reading inputs...')
    if args.input_format == 'matlab':
        with h5py.File(args.input_file, 'r') as f:
            bags = read_cell_array(f, f[args.bags_name])
            cats = np.squeeze(f['cats'][...])
            if args.labels_name:
                labels = np.squeeze(f[args.labels_name][...])
            else:
                labels = cats
    else:
        assert args.input_format == 'python'

        if os.path.isdir(args.input_file):
            from .image_features import read_features_perimage
            feats = read_features_perimage(args.input_file)
        else:
            from .image_features import read_features
            feats = read_features(args.input_file)
        bags = feats.features
        names = np.asarray(feats.names)
        cats = np.asarray(feats.categories)

        if args.labels_name:
            labels = np.array([ex[args.labels_name] for ex in feats.extras])
        elif args.labels_name is None and args.svm_mode == 'SVC':
            labels = cats
        else:
            raise ValueError("must provide a label name when regressing")

        del feats

    if args.svm_mode == 'SVC' and not is_categorical_type(labels):
        if labels.dtype.kind == 'f' and np.all(labels == np.round(labels)):
            labels = labels.astype(int)
        else:
            label_names = labels
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(label_names)

    if args.n_points:
        bags = subset_data(bags, args.n_points)

    opts = opts_dict(args)
    opts['cats'] = cats
    if args.input_format == 'python':
        opts['names'] = names
    acc, preds = crossvalidate(bags, labels,
        num_folds=args.cv_folds, stratified_cv=args.stratified_cv,
        divs_cache=args.div_cache_file, **opts)

    status_fn('')
    if args.svm_mode == 'SVC':
        status_fn('Accuracy: {:.1%}'.format(acc))
    elif args.svm_mode == 'NuSVR':
        status_fn('RMSE: {}'.format(acc))

    out = {
        'div_func': args.div_func,
        'K': args.K,
        'C_vals': args.c_vals,
        'sigma_vals': args.sigma_vals,
        'scale_sigma': args.scale_sigma,
        'preds': preds,
        'svm_mode': args.svm_mode,
    }
    if 'svm_mode' == 'SVC':
        out['acc'] = acc
        if args.input_format != 'matlab':
            out['label_names'] = label_encoder.classes_
    elif args.svm_mode == 'NuSVR':
        out['rmse'] = acc
        out['svr_nu_vals'] = args.svr_nu_vals
    status_fn('Saving output to {}'.format(args.output_file))
    scipy.io.savemat(args.output_file, out, oned_as='column')


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
