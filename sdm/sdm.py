#!/usr/bin/env python
'''
Code to do classification with divergence kernels in SVMs, as in
    Dougal J. Sutherland, Liang Xiong, Barnabas Poczos, Jeff Schneider.
    Kernels on Sample Sets via Nonparametric Divergence Estimates.
    http://arxiv.org/abs/1202.0302
'''

from __future__ import division, print_function

from collections import Counter
from functools import partial, reduce
import itertools
from operator import itemgetter, mul
import os
import random
import sys
import warnings

import h5py
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.base
from sklearn.cross_validation import KFold, StratifiedKFold
try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    from sklearn.grid_search import IterGrid as ParameterGrid
from sklearn.metrics import accuracy_score, zero_one_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ConvergenceWarning
from sklearn import svm  # NOTE: needs version 0.13+ for svm iter limits

from .utils import (positive_int, positive_float, portion, is_integer_type,
                    iteritems, izip)
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
DEFAULT_SYMMETRIZE_DIVS = False

def get_status_fn(val):
    if val is True:
        return partial(print, file=sys.stderr)
    elif val is None:
        return lambda *args, **kwargs: None
    else:
        return val


def is_categorical_type(ary):
    return is_integer_type(ary) or np.asanyarray(ary).dtype.kind == 'b'


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


################################################################################
### PSD projection and friends


def symmetrize(mat, destroy=False):
    '''
    Returns the mean of mat and its transpose.

    If destroy, invalidates the passed-in matrix.
    '''
    # TODO: figure out a no-copy version of this that actually works...
    #       might have to write it in cython. probably not worth it.
    mat = mat + mat.T
    mat /= 2
    return mat


def project_psd(mat, min_eig=0, destroy=False, negatives_likely=True):
    '''
    Project a real symmetric matrix to PSD by discarding any negative
    eigenvalues from its spectrum. Passing min_eig > 0 lets you similarly make
    it positive-definite, though this may not technically be a projection...?

    Symmetrizes the matrix before projecting.

    If destroy is True, invalidates the passed-in matrix.

    If negatives_likely (default), optimizes memory usage for the case where we
    expect there to be negative eigenvalues.
    '''
    mat = symmetrize(mat, destroy=destroy)

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat, overwrite_a=negatives_likely)
    if negatives_likely or vecs.min() < min_eig:
        del mat
        np.maximum(vals, min_eig, vals)  # update vals in-place
        mat = np.dot(vecs, vals.reshape(-1, 1) * vecs.T)
        del vals, vecs
        mat = symmetrize(mat, destroy=True)  # should be symmetric, but do it
                                             # anyway for numerical reasons
    return mat


def rbf_kernelize(divs, sigma, destroy=False):
    '''
    Passes a distance matrix through an RBF kernel.

    If destroy, does it in-place.
    '''
    if destroy:
        km = divs
        km **= 2
    else:
        km = divs ** 2
    # TODO do we want to square, say, Renyi divergences?
    km /= -2 * sigma**2
    np.exp(km, km)  # inplace
    return km


def make_km(divs, sigma, destroy=False, negatives_likely=True):
    '''
    Passes a distance matrix through an RBF kernel and then discards any
    negative eigenvalues.

    If destroy, invalidates the data in divs.

    If negatives_likely (default), optimizes memory usage for the case where we
    expect there to be negative eigenvalues.
    '''
    return project_psd(rbf_kernelize(divs, sigma, destroy=destroy),
                       destroy=True, negatives_likely=negatives_likely)


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
### Main dealio


def _try_params(cls, tuning_params, sigma_kms, labels, folds, svm_params):
    params = tuning_params.copy()

    train_idx, test_idx = folds.value[params.pop('fold_idx')]
    train_km, test_km = split_km(
        sigma_kms[params.pop('sigma')].value, train_idx, test_idx)

    params.update(svm_params)
    clf = cls.svm_class(**params)
    clf.fit(train_km, labels.value[train_idx])

    preds = clf.predict(test_km)
    assert not np.any(np.isnan(preds))
    loss = cls.tuning_loss(labels.value[test_idx], preds)
    return tuning_params, loss, clf.fit_status_


class BaseSDM(sklearn.base.BaseEstimator):
    # TODO: support non-Gaussian kernels
    # TODO: support CVing between multiple div funcs, values of K
    # TODO: support more SVM options
    # TODO: change API to be nicer to just give divs/km w/o faking data
    def __init__(self,
                 div_func='renyi:.9',
                 K=DEFAULT_K,
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
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 save_bags=True):
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
        self.symmetrize_divs = symmetrize_divs
        self.save_bags = save_bags

    @property
    def classifier(self):
        raise NotImplementedError

    @property
    def regressor(self):
        raise NotImplementedError

    @property
    def tuning_loss(self, true, pred):
        raise NotImplementedError

    @property
    def _show_tuning_choices(self):
        raise NotImplementedError

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

        if self.symmetrize_divs:
            divs = symmetrize(divs)

        # tune params
        self.status_fn('Tuning SVM parameters...')
        self._tune_params(
                divs=np.ascontiguousarray(divs[np.ix_(train_idx, train_idx)]),
                labels=train_y)

        # project the final Gram matrix
        self.status_fn('Doing final projection')
        full_km = make_km(divs, self.sigma_)
        train_km = np.ascontiguousarray(full_km[np.ix_(train_idx, train_idx)])

        # train the selected SVM
        self.status_fn('Training final SVM')
        params = self._svm_params(tuning=False)
        clf = self.svm_class(**params)
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
            km = rbf_kernelize(divs, self.sigma_, destroy=False)
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
            km = rbf_kernelize(divs, self.sigma_, destroy=True)

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

    ############################################################################
    ### Parameter tuning
    def _param_grid_dict(self):
        return {
            'sigma': np.sort(self.sigma_vals),
            'C': np.sort(self.C_vals),
        }

    def _set_tuning(self, d):
        for k, v in iteritems(d):
            setattr(self, k + '_', v)

    def _svm_params(self, tuning=False):
        d = {
            'cache_size': self.tuning_cache_size if tuning else self.cache_size,
            'kernel': 'precomputed',
            'tol': self.tuning_svm_tol if tuning else self.svm_tol,
            'max_iter': self.tuning_svm_max_iter if tuning else self.svm_max_iter,
            'shrinking': self.svm_shrinking,
            # 'verbose': False,
        }
        if not tuning:
            d['C'] = self.C_
        return d

    def _tune_params(self, divs, labels):
        # check input shapes
        num_folds = self.tuning_folds
        num_bags = divs.shape[0]
        if not (divs.ndim == 2 and divs.shape[1] == num_bags):
            msg = "divs is {}, should be ({1},{1})".format(divs.shape, num_bags)
            raise ValueError("divs is wrong shape")
        if labels.shape != (num_bags,):
            msg = "labels is {}, should be ({},)".format(labels.shape, num_bags)
            raise ValueError(msg)

        # figure out the hypergrid of parameter options
        param_d = self._param_grid_dict()
        if self.scale_sigma:
            param_d['sigma'] = param_d['sigma'] * np.median(divs[divs > 0])
            # make sure not to modify self.sigma_vals...
        folds = ForkedData(
                    list(KFold(n=num_bags, n_folds=num_folds, shuffle=True)))
        param_d['fold_idx'] = np.arange(num_folds)
        param_names, param_lens = zip(*sorted(
                (name, len(vals)) for name, vals in iteritems(param_d)))
        param_grid = ParameterGrid(param_d)

        # do we have multiple options to try?
        num_pts = reduce(mul, param_lens)
        if num_pts == 0:
            raise ValueError("no parameters in tuning grid")
        elif num_pts == num_folds:  # only one param set, no tuning necessary
            self._set_tuning(next(param_grid))
            return

        # get kernel matrices for the sigma vals we're trying
        # TODO: could be more careful about making copies here
        sigma_kms = {}
        self.status_fn('Projecting...')
        for sigma in param_d['sigma']:
            #status_fn('Projecting: sigma = {}'.format(sigma))
            sigma_kms[sigma] = ForkedData(make_km(divs, sigma))

        labels_d = ForkedData(labels)

        ### try each param combination and see how they do
        self.status_fn('Cross-validating parameter sets...')

        # make the hypergrid and fill in tuning loss
        scores = np.empty(tuple(param_lens))
        scores.fill(np.nan)

        if self.progressbar:
            pbar, tick_pbar = progressbar_and_updater(maxval=scores.size)

        # filter convergence warnings, count them up instead
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        ignore_conv = warnings.filters[0]
        conv_warning_counter = itertools.count()

        # function that gets loss for a given set of params
        try_params = partial(_try_params, self.__class__,
                             sigma_kms=sigma_kms, labels=labels_d, folds=folds,
                             svm_params=self._svm_params(tuning=True))

        # callback to save the resulting score
        def assign_score(params_val_and_status):
            params, val, status = params_val_and_status
            indices = tuple(param_d[k].searchsorted(params[k])
                            for k in param_names)
            scores[indices] = val
            if status:
                next(conv_warning_counter)
            if self.progressbar:
                tick_pbar()

        # actually do it
        with get_pool(self.n_proc) as pool:
            for job in param_grid:
                pool.apply_async(try_params, [job], callback=assign_score)

        if self.progressbar:
            pbar.finish()

        warnings.filters.remove(ignore_conv)
        self.status_fn('{} SVMs terminated early, after {:,} steps'.format(
            next(conv_warning_counter), self.tuning_svm_max_iter))

        # figure out which ones were best
        assert not np.any(np.isnan(scores))
        fold_idx_idx = param_names.index('fold_idx')
        nonfold_param_names = (
                param_names[:fold_idx_idx] + param_names[fold_idx_idx+1:])

        cv_means = scores.mean(axis=fold_idx_idx)
        best_elts = cv_means == cv_means.min()
        best_indices = random.choice(np.transpose(best_elts.nonzero()))
        the_params = dict(
                (name, param_d[name][idx])
                for name, idx in izip(nonfold_param_names, best_indices))
        self._set_tuning(the_params)

    ############################################################################
    ### Cross-validation helper
    def crossvalidate(self, bags, labels,
            num_folds=10, stratified_cv=False,
            project_all=True,
            divs=None, divs_cache=None, names=None, cats=None):
        # TODO: allow specifying what the folds should be
        # TODO: optionally return params for each fold, what the folds were, ...
        status = self.status_fn

        num_bags = len(bags)
        dim = bags[0].shape[1]
        if any(bag.ndim != 2 or bag.shape[1] != dim or bag.shape[0] == 0
               for bag in bags):
            msg = "all bags should be n_i x dim, with n_i > 0, consistent dim"
            raise ValueError(msg)

        labels = np.squeeze(labels)
        assert labels.shape == (num_bags,)
        if self.classifier:
            assert is_categorical_type(labels)
            assert np.all(labels >= 0)
        else:
            assert np.all(np.isfinite(labels))

        if divs is None:
            status('Getting divergences...')
            divs = get_divs_cache(bags,
                    div_func=self.div_func, K=self.K,
                    cache_filename=divs_cache, n_proc=self.n_proc,
                    fix_mode=self.fix_mode, tail=self.tail,
                    min_dist=self.min_dist,
                    names=names, cats=cats,
                    status_fn=self._status_fn, progressbar=self.progressbar)
        else:
            #status_fn('Using passed-in divergences...')
            assert divs.shape == (num_bags, num_bags)

        if self.classifier:
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

            if self.classifier:
                status('Test distribution: {}'.format(dict(Counter(labels[test]))))

            if project_all:
                preds[test] = self.transduct(
                        train_bags, labels[train], test_bags,
                        divs=divs[np.ix_(both, both)])
            else:
                self.fit(train_bags, labels[train],
                         divs=divs[np.ix_(train, train)])
                preds[test] = self.predict(test_bags)

            score = self.eval_score(labels[test], preds[test])
            status('Fold {score_name}: {score:{score_fmt}}'.format(score=score,
                        score_name=self.score_name, score_fmt=self.score_fmt))

        return self.eval_score(labels, preds), preds


class SDC(BaseSDM):
    classifier = True
    regressor = False
    svm_class = svm.SVC
    tuning_loss = staticmethod(zero_one_loss)
    eval_score = staticmethod(accuracy_score)
    score_name = 'accuracy'
    score_fmt = '.1%'

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
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 status_fn=None, progressbar=None,
                 fix_mode=FIX_MODE_DEFAULT, tail=TAIL_DEFAULT, min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 save_bags=True):
        super(SDC, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            C_vals=C_vals, sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_shrinking=svm_shrinking,
            status_fn=status_fn, progressbar=progressbar,
            fix_mode=fix_mode, tail=tail, min_dist=min_dist,
            symmetrize_divs=symmetrize_divs, save_bags=save_bags)
        self.weight_classes = weight_classes

    def _svm_params(self, tuning=False):
        d = super(SDC, self)._svm_params(tuning=tuning)
        d['class_weight'] = 'auto' if self.weight_classes else None
        return d

    def _show_tuning_choices(self):
        self.status_fn('Chose sigma {o.sigma_}, C {o.C_}, nu {o.svr_nu_}'
                       .format(o=self))


class NuSDR(BaseSDM):
    classifier = False
    regressor = True
    svm_class = svm.NuSVR
    tuning_loss = staticmethod(mean_squared_error)
    eval_score = staticmethod(rmse)
    score_name = 'MSE'
    score_fmt = ''

    def __init__(self,
                 div_func='renyi:.9',
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 C_vals=DEFAULT_C_VALS,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 svr_nu_vals=DEFAULT_SVR_NU_VALS,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 status_fn=None, progressbar=None,
                 fix_mode=FIX_MODE_DEFAULT, tail=TAIL_DEFAULT, min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 save_bags=True):
        super(NuSDR, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            C_vals=C_vals, sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_shrinking=svm_shrinking,
            status_fn=status_fn, progressbar=progressbar,
            fix_mode=fix_mode, tail=tail, min_dist=min_dist,
            symmetrize_divs=symmetrize_divs, save_bags=save_bags)
        self.svr_nu_vals = svr_nu_vals

    def _param_grid_dict(self):
        d = super(NuSDR, self)._param_grid_dict()
        d['nu'] = np.sort(self.svr_nu_vals)
        return d

    def _svm_params(self, tuning=False):
        d = super(NuSDR, self)._svm_params(tuning=tuning)
        if not tuning:
            d['nu'] = self.svr_nu_
        return d

    def _set_tuning(self, d):
        self.svr_nu_ = d.pop('nu')
        super(NuSDR, self)._set_tuning(d)

    def _show_tuning_choices(self):
        self.status_fn('Chose sigma {o.sigma_}, C {o.C_}'.format(o=self))


sdm_for_mode = {
    'SVC': SDC,
    'NuSVR': NuSDR,
}

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

        algo._add_action(ActionNoYes('symmetrize-divs', default=False,
            help="Symmetrize divergence estimates before passing them through "
                 "the RBF kernel, rather than after."))

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
    d = {
        'div_func': args.div_func,
        'K': args.K,
        'tuning_folds': args.tuning_folds,
        'n_proc': args.n_proc,
        'C_vals': args.c_vals,
        'sigma_vals': args.sigma_vals, 'scale_sigma': args.scale_sigma,
        'cache_size': args.cache_size,
        'tuning_cache_size': args.tuning_cache_size,
        'svm_tol': args.svm_tol,
        'tuning_svm_tol': args.tuning_svm_tol,
        'svm_max_iter': args.svm_max_iter,
        'tuning_svm_max_iter': args.tuning_svm_max_iter,
        'svm_shrinking': args.svm_shrinking,
        'tail': args.trim_tails,
        'fix_mode': args.trim_mode,
        'min_dist': args.min_dist,
        'symmetrize_divs': args.symmetrize_divs,
    }
    if args.svm_mode == 'SVC':
        d['weight_classes'] = args.weight_classes
    elif args.svm_mode == 'NuSVR':
        d['svr_nu_vals'] = args.svr_nu_vals
    return d


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

    clf = SDC(status_fn=True, **opts_dict(args))
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
            names = None
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

    label_class_names = None
    if args.svm_mode == 'SVC' and not is_categorical_type(labels):
        if labels.dtype.kind == 'f' and np.all(labels == np.round(labels)):
            labels = labels.astype(int)
        else:
            label_names = labels
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(label_names)
            label_class_names = label_encoder.classes_

    if args.n_points:
        bags = subset_data(bags, args.n_points)

    clf = sdm_for_mode[args.svm_mode](status_fn=True, **opts_dict(args))
    score, preds = clf.crossvalidate(bags, labels,
        num_folds=args.cv_folds, stratified_cv=args.stratified_cv,
        project_all=args.mode == 'transduct',
        divs_cache=args.div_cache_file, cats=cats, names=names)

    status_fn('')
    if args.svm_mode == 'SVC':
        status_fn('Accuracy: {:.1%}'.format(score))
    elif args.svm_mode == 'NuSVR':
        status_fn('RMSE: {}'.format(score))

    out = {
        'div_func': args.div_func,
        'K': args.K,
        'C_vals': args.c_vals,
        'sigma_vals': args.sigma_vals,
        'scale_sigma': args.scale_sigma,
        'preds': preds,
        'labels': labels,
        'svm_mode': args.svm_mode,
    }
    if args.svm_mode == 'SVC':
        out['acc'] = acc
        if label_encoder is not None:
            out['label_names'] = label_class_names
    elif args.svm_mode == 'NuSVR':
        out['rmse'] = score
        out['svr_nu_vals'] = args.svr_nu_vals
    status_fn('Saving output to {}'.format(args.output_file))
    scipy.io.savemat(args.output_file, out, oned_as='column')


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
