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
from operator import mul
import os
import random
import warnings

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

from .features import Features
from .utils import (positive_int, positive_float, portion,
                    is_categorical_type,
                    rmse, iteritems, iterkeys, izip, identity, lazy_range,
                    get_status_fn, read_cell_array)
from .mp_utils import ForkedData, get_pool, progressbar_and_updater
from .np_divs import (estimate_divs,
                      check_h5_settings, add_to_h5_cache, normalize_div_name)


# TODO: better logging

DEFAULT_SVM_CACHE = 1000
DEFAULT_SVM_TOL = 1e-3
DEFAULT_SVM_ITER = 10 ** 6
DEFAULT_SVM_ITER_TUNING = 1000
DEFAULT_SVM_SHRINKING = True
DEFAULT_SVM_PROBABILITY = False

DEFAULT_DIV_FUNC = 'kl'
DEFAULT_SIGMA_VALS = tuple(2.0 ** np.arange(-4, 11, 2))
DEFAULT_C_VALS = tuple(2.0 ** np.arange(-9, 19, 3))
DEFAULT_NU_VALS = (0.1, 0.2, 0.3, 0.5, 0.7)
DEFAULT_SVR_EPSILON_VALS = tuple(10.0 ** np.arange(-3, 5))
DEFAULT_SVR_NU_VALS = (0.2, 0.3, 0.5, 0.7)
DEFAULT_K = 3
DEFAULT_TUNING_FOLDS = 3
DEFAULT_SYMMETRIZE_DIVS = False
DEFAULT_KM_METHOD = 'clip'
DEFAULT_TRANSFORM_TEST = True


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


def _transformer(transform, test_matrix):
    '''
    Applies a given transformation matrix to the matrix of test vector
    similarities (num_test x num_train).
    '''
    return np.dot(transform, test_matrix.T).T


def project_psd(mat, min_eig=0, destroy=False, negatives_likely=True,
                ret_test_transformer=False):
    '''
    Project a real symmetric matrix to PSD by discarding any negative
    eigenvalues from its spectrum. Passing min_eig > 0 lets you similarly make
    it positive-definite, though this may not technically be a projection...?

    Symmetrizes the matrix before projecting.

    If destroy is True, invalidates the passed-in matrix.

    If negatives_likely (default), optimizes for the case where we expect there
    to be negative eigenvalues.

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent. Uses the method of
       Chen, Y., Garcia, E. K., Gupta, M. R., Rahimi, A., & Cazzanti, L. (2009).
       Similarity-based classification: Concepts and algorithms.
       Journal of Machine Learning Research, 10, 747-776.
    '''
    mat = symmetrize(mat, destroy=destroy)

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat, overwrite_a=negatives_likely)
    vals = vals.reshape(-1, 1)

    if ret_test_transformer:
        clip = np.dot(vecs, (vals > 0) * vecs.T)
        transform = partial(_transformer, clip)

    if negatives_likely or vals[0, 0] < min_eig:
        del mat
        np.maximum(vals, min_eig, vals)  # update vals in-place
        mat = np.dot(vecs, vals.reshape(-1, 1) * vecs.T)
        if not ret_test_transformer:
            del vals, vecs
        mat = symmetrize(mat, destroy=True)  # should be symmetric, but do it
                                             # anyway for numerical reasons

    if ret_test_transformer:
        return mat, transform
    return mat


def shift_psd(mat, min_eig=0, destroy=False, negatives_likely=True,
              ret_test_transformer=False):
    '''
    Turn a real symmetric matrix to PSD by adding to its diagonal. Passing
    min_eig > 0 lets you make it positive-definite.

    Symmetrizes the matrix before doing so.

    If destroy is True, modifies the passed-in matrix in-place.

    Ignores the negatives_likely argument (just there for consistency).

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent. For the shift method, which only affects
    self-similarities, this is just the identity function.
    '''
    mat = symmetrize(mat, destroy=destroy)
    lo, = scipy.linalg.eigvalsh(mat, eigvals=(0, 0))
    diff = min_eig - lo
    if diff < 0:
        r = lazy_range(mat.shape[0])
        mat[r, r] += diff

    if ret_test_transformer:
        return mat, identity
    return mat


def flip_psd(mat, destroy=False, negatives_likely=True,
             ret_test_transformer=False):
    '''
    Turn a real symmetric matrix into PSD by flipping the sign of any negative
    eigenvalues in its spectrum.

    If destroy is True, invalidates the passed-in matrix.

    If negatives_likely (default), optimizes for the case where we expect there
    to be negative eigenvalues.

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent. Uses the method of
       Chen, Y., Garcia, E. K., Gupta, M. R., Rahimi, A., & Cazzanti, L. (2009).
       Similarity-based classification: Concepts and algorithms.
       Journal of Machine Learning Research, 10, 747-776.
    '''
    mat = symmetrize(mat, destroy=destroy)

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat, overwrite_a=negatives_likely)
    vals = vals.reshape(-1, 1)

    if ret_test_transformer:
        flip = np.dot(vecs, np.sign(vals) * vecs.T)
        transform = partial(_transformer, flip)

    if negatives_likely or vals[0, 0] < 0:
        del mat
        np.abs(vals, vals)  # update vals in-place
        mat = np.dot(vecs, vals * vecs.T)
        del vals, vecs
        mat = symmetrize(mat, destroy=True)  # should be symmetric, but do it
                                             # anyway for numerical reasons
    if ret_test_transformer:
        return mat, transform
    return mat


def square_psd(mat, destroy=False, negatives_likely=True,
               ret_test_transformer=False):
    '''
    Turns a real matrix into a symmetric psd one through S -> S S^T. Equivalent
    to squaring the eigenvalues in a spectral decomposition, or to using the
    similarities to test points as features in a linear classifier.

    Ignores the destroy and negatives_likely arguments (just there for
    consistency).

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent.
    '''
    if ret_test_transformer:
        # TODO: do this like a linear operator, or using sims as features, or...
        raise NotImplementedError("not sure how to transform test samples here")
    return np.dot(mat, mat.T)


def identity_psd(mat, destroy=False, negatives_likely=True,
                 ret_test_transformer=False):
    if ret_test_transformer:
        return mat, np.eye(mat.shape[0])
    return mat


psdizers = {
    'project': project_psd,
    'clip': project_psd,
    'shift': shift_psd,
    'flip': flip_psd,
    'square': square_psd,
    'identity': identity_psd,
}


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


def make_km(divs, sigma, destroy=False, negatives_likely=True,
            method=DEFAULT_KM_METHOD, ret_test_transformer=False):
    '''
    Passes a distance matrix through an RBF kernel of bandwidth sigma, and then
    ensures that it's PSD through `method` (see `psdizers`). Default: projects
    to the nearest PSD matrix by clipping any negative eigenvalues.

    If destroy, invalidates the data in divs.

    If negatives_likely (default), optimizes memory usage for the case where we
    expect there to be negative eigenvalues.
    '''
    return psdizers[method](
        rbf_kernelize(divs, sigma, destroy=destroy),
        destroy=True,
        negatives_likely=negatives_likely,
        ret_test_transformer=ret_test_transformer)


def split_km(km, train_idx, test_idx):
    train_km = np.ascontiguousarray(km[np.ix_(train_idx, train_idx)])
    test_km = np.ascontiguousarray(km[np.ix_(test_idx, train_idx)])
    return train_km, test_km


################################################################################
### Cached divs helper

def get_divs_cache(bags, div_func, K, cache_filename=None, min_dist=None,
                   n_proc=None, status_fn=True, progressbar=None):
    import h5py
    # TODO: support loading subsets of the file, or reordered, based on names
    # TODO: support flann arguments

    status = get_status_fn(status_fn)

    if cache_filename and os.path.exists(cache_filename):
        path = '{}/{}'.format(div_func, K)
        with h5py.File(cache_filename, 'r') as f:
            check_h5_settings(f, n=len(bags), dim=bags.dim,
                min_dist=min_dist,
                names=bags.names, cats=bags.categories)
            if path in f:
                divs = f[path]
                # assert divs.shape == (len(bags), len(bags)) # in check

                status("Loading divs from cache '{}'".format(cache_filename))
                return divs[...]

    divs = np.squeeze(estimate_divs(
            bags, specs=[div_func], Ks=[K],
            cores=n_proc, min_dist=min_dist,
            status_fn=status_fn, progressbar=progressbar))

    if cache_filename:
        status("Saving divs to cache '{}'".format(cache_filename))
        with h5py.File(cache_filename) as f:
            add_to_h5_cache(f, {(div_func, K): divs},
                            dim=bags.dim,
                            min_dist=min_dist,
                            names=bags.names, cats=bags.categories)

    return divs


################################################################################
### Main dealio


def _try_params(cls, tuning_params, sigma_kms, labels, folds, svm_params,
                sample_weight=None):
    params = tuning_params.copy()

    train_idx, test_idx = folds.value[params.pop('fold_idx')]
    train_km, test_km = split_km(
        sigma_kms[params.pop('sigma')].value, train_idx, test_idx)

    params.update(svm_params)
    clf = cls.svm_class(**params)

    opts = {}
    if sample_weight is not None and sample_weight.value is not None:
        opts['sample_weight'] = sample_weight.value[train_idx]

    try:
        clf.fit(train_km, labels.value[train_idx], **opts)
        preds = clf.predict(test_km)
        assert not np.any(np.isnan(preds))
        loss = cls.tuning_loss(labels.value[test_idx], preds)
        status = 'convergence warning' if clf.fit_status_ else None
        return tuning_params, loss, status
    except ValueError as e:
        return tuning_params, 1e50, e.args
        # using 1e50 because if *everything* errors, want to get the one that
        # failed the least often
        # TODO: count these like we count the fit_status_ errors


def _not_implemented(*args, **kwargs):
    raise NotImplementedError


class BaseSDM(sklearn.base.BaseEstimator):
    # TODO: support squaring or not squaring divs
    # TODO: support non-Gaussian kernels
    # TODO: support CVing between multiple div funcs, values of K
    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
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
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        self.div_func = div_func
        self.K = K
        self.tuning_folds = tuning_folds
        self.n_proc = n_proc
        self.sigma_vals = sigma_vals
        self.scale_sigma = scale_sigma
        self.weight_classes = weight_classes
        self.cache_size = cache_size
        self.tuning_cache_size = tuning_cache_size
        self.svm_tol = svm_tol
        self.tuning_svm_tol = tuning_svm_tol
        self.svm_max_iter = svm_max_iter
        self.tuning_svm_max_iter = tuning_svm_max_iter
        self.svm_shrinking = svm_shrinking
        self.status_fn = status_fn
        self.progressbar = progressbar
        self.min_dist = min_dist
        self.symmetrize_divs = symmetrize_divs
        self.km_method = km_method
        self.transform_test = transform_test
        self.save_bags = save_bags

    classifier = False
    regressor = False
    oneclass = False
    svm_class = property(_not_implemented)
    tuning_loss = staticmethod(_not_implemented)
    eval_score = staticmethod(_not_implemented)
    score_name = property(_not_implemented)
    score_fmt = property(_not_implemented)

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

    def _div_args(self, for_cache=True):
        d = {
            'min_dist': self.min_dist,
            'status_fn': self._status_fn,
            'progressbar': self.progressbar,
        }
        if for_cache:
            d.update({
                'div_func': self.div_func,
                'K': self.K,
                'n_proc': self.n_proc,
            })
        else:
            d.update({
                'specs': [self.div_func],
                'Ks': [self.K],
                'cores': self.n_proc,
            })
        return d

    def clear_fit(self):
        for attr_name in dir(self):
            if attr_name.endswith('_') and not attr_name.startswith('_'):
                delattr(self, attr_name)

    _fit_docstr = '''
        Tunes parameters and then fits an SVM using the final parameters
        on training data.

        X should be one of:
            - an sdm.Features instance (preferred)
            - a list of row-instance data matrices each (num_pts x dim),
              where num_pts can vary but dim needs to be constant
            - None, if save_bags is False and you're passing divs. In this case,
              you need to compute divergences for any test points yourself.
        {y_doc}
        sample_weight (optional): a vector of weights applied to each sample,
            where 1 means unweighted.

        divs (optional): precomputed divergences among the passed points
            (an array of shape num_bags x num_bags)

        divs_cache (optional): a filename for a cache file. Note that this
            needs to be on the TRAINING data only, in the same order. Any names
            or categories in X are used to verify the cache file is for the
            right data.

        ret_km (optional, boolean): if True, returns the final training kernel
            matrix.
    '''
    def fit(self, X, y, sample_weight=None, divs=None, divs_cache=None,
            ret_km=False):
        if X is None:
            if divs is None:
                raise ValueError("need to pass either X or divs to fit()")
            if self.save_bags:
                msg = "Need to pass data to fit() if save_bags is true."
                raise ValueError(msg)
            n_bags = None
        else:
            n_bags = len(X)

        if divs is not None:
            divs = np.asarray(divs)
            if divs.ndim != 2:
                raise ValueError("divs should be n_bags x n_bags")
            a, b = divs.shape
            if a != b:
                raise ValueError("divs should be n_bags x n_bags")

            if n_bags is None:
                n_bags = a
            elif a != n_bags:
                raise ValueError("divs should be n_bags x n_bags")

        y = np.squeeze(y)
        if y.shape != (n_bags,):
            raise ValueError("y should be 1d of length n_bags")

        if self.classifier:
            if not is_categorical_type(y) or np.any(y < -1):
                raise ValueError("y for classification should be ints >= -1")
            train_idx = y != -1
        else:
            train_idx = ~np.isnan(y)

        if sample_weight is None:
            train_sample_weight = None
        else:
            sample_weight = np.asarray(sample_weight)
            train_sample_weight = sample_weight[train_idx]

        train_y = y[train_idx]
        if train_y.size < 2:
            raise ValueError("must train with at least 2 points")
        if not self.oneclass and np.all(train_y == train_y[0]):
            raise ValueError("can't train with only one class")

        if (self.save_bags or divs is None) and not isinstance(X, Features):
            X = Features(X)

        if self.save_bags:
            self.train_bags_ = X[train_idx]

        # get divergences
        if divs is None:
            self.status_fn('Getting divergences...')
            div_args = self._div_args(for_cache=True)
            divs = get_divs_cache(X, cache_filename=divs_cache, **div_args)

        if self.symmetrize_divs:
            divs = symmetrize(divs)

        # tune params
        self.status_fn('Tuning SVM parameters...')
        self._tune_params(
                divs=np.ascontiguousarray(divs[np.ix_(train_idx, train_idx)]),
                labels=train_y, sample_weight=train_sample_weight)

        # project the final Gram matrix
        self.status_fn('Doing final projection')
        fn = partial(make_km, divs, self.sigma_, method=self.km_method)
        if self.transform_test:
            full_km, self.test_transformer_ = fn(ret_test_transformer=True)
        else:
            full_km = fn()
        train_km = np.ascontiguousarray(full_km[np.ix_(train_idx, train_idx)])

        # train the selected SVM
        self.status_fn('Training final SVM')
        params = self._svm_params(tuning=False)
        clf = self.svm_class(**params)
        if self.oneclass:
            clf.fit(train_km, sample_weight=sample_weight)
        else:
            clf.fit(train_km, train_y, sample_weight=sample_weight)
        self.svm_ = clf

        if ret_km:
            return full_km
    fit.__doc__ = _fit_docstr.format(y_doc="""
        y: a vector of class labels (depending on the subclass)
        """)

    def _prediction_km(self, data=None, divs=None, km=None):
        # TODO: smarter projection options for inductive use
        if getattr(self, 'svm_', None) is None:
            raise ValueError("SDM: need to fit before you can predict!")

        if km is not None:
            return km

        if divs is not None:
            destroy_divs = False
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
                    self.train_bags_ + data, mask=mask,
                    **self._div_args(for_cache=False)))
            divs = (divs[-n_test:, :n_train] + divs[:n_train, -n_test].T) / 2
            destroy_divs = True

        km = rbf_kernelize(divs, self.sigma_, destroy=destroy_divs)
        if self.transform_test:
            km = self.test_transformer_(km)
        return km

    def predict(self, data, divs=None, km=None):
        km = self._prediction_km(data, divs=divs, km=km)

        preds = self.svm_.predict(km)
        if self.classifier:
            assert np.all(preds == np.round(preds))
            return preds.astype(int)
        else:
            return preds

    def decision_function(self, data, divs=None, km=None):
        km = self._prediction_km(data, divs=divs, km=km)
        return self.svm_.decision_function(km)

    def score(self, data, labels, divs=None, km=None):
        preds = self.predict(data, divs=divs, km=km)
        return self.eval_score(labels, preds)

    def transduct(self, train_bags, train_labels, test_bags, divs=None,
                  mode='predict', save_fit=False):
        '''
        Trains an SDM transductively, where the kernel matrix is constructed on
        the training + test points, the SVM is trained on training points, and
        predictions are done on the test points.

        The SVM itself is inductive (given the kernel).

        Arguments
        ---------

        train_bags: a Features instance, list of row-instance data matrices,
            or None. (If None, divs is required.)

        train_labels: a label vector, like y for fit() except that the
            "semi-supervised" label is not supported.

        test_bags: a Features instance, list of row-instance data matrices,
            or None. (If None, divs is required.)

        divs (optional): a matrix of divergences of shape
            (num_train + num_test, num_train + num_test), ordered with the
            training bags first and then the test bags following.
            Transparent caching is not yet supported here.

        mode (default 'predict'): one of 'predict', 'dec', 'proba', 'log_proba'.
            Returns the results of predict(), decision_function(),
            predict_proba(), or predict_log_proba(), depending on this argument.
            The latter two only work for classifiers.

        save_fit (boolean, default false): By default, the SDM object does not
            save its fit and is reset to an un-fit state as if it had just been
            constructed. Passing save_fit=True makes the fit persistent.
        '''
        # TODO: support transparent divs caching by passing in indices
        # TODO: support passing in pre-stacked train/test features,
        #       for minimal stacking purposes

        train_labels = np.squeeze(train_labels)
        if train_labels.ndim != 1:
            raise TypeError("train_labels should be 1d")
        n_train = train_labels.shape[0]

        pred_fns = {'predict': self.predict, 'dec': self.decision_function}
        if self.classifier:
            pred_fns['proba'] = self.predict_proba
            pred_fns['log_proba'] = self.predict_log_proba
        if mode not in pred_fns:
            raise ValueError("unknown transduction mode '{}'".format(mode))
        pred_fn = pred_fns[mode]

        if self.classifier:
            if not is_categorical_type(train_labels) or train_labels.min() < 0:
                raise TypeError("train_labels should be nonnegative integers")
        else:
            if not np.all(np.isfinite(train_labels)):
                raise TypeError("train_labels should be finite")

        if divs is not None:
            divs = np.asarray(divs)
            if divs.ndim != 2:
                raise TypeError("divs should be 2d")
            n_both, b = divs.shape
            if n_both != b:
                raise TypeError("divs should be n_bags x n_bags")
            n_test = n_both - n_train
            if n_test < 1:
                raise TypeError("divs should have length num_train + num_test")
            combo_bags = None
        else:
            if train_bags is None or test_bags is None:
                raise TypeError("must pass either divs or train_bags/test_bags")

            if len(train_bags) != n_train:
                raise TypeError("train_bags and train_labels should have "
                                "consistent lengths")
            n_test = len(test_bags)

        if divs is None or (save_fit and self.save_bags):
            if train_bags is None or test_bags is None:
                raise TypeError("must pass bags if save_fit and save_bags")

            combo_bags = train_bags + test_bags
            if not isinstance(combo_bags, Features):
                combo_bags = Features(combo_bags)

        if not save_fit:
            old_save_bags = self.save_bags
            self.save_bags = False  # avoid keeping copies around

        # make fake labels for test data, so fit() knows what they are
        test_fake_labels = np.empty(n_test, dtype=train_labels.dtype)
        test_fake_labels.fill(-1 if self.classifier else np.nan)
        combo_labels = np.hstack((train_labels, test_fake_labels))

        full_km = self.fit(combo_bags, combo_labels, divs=divs, ret_km=True)
        preds = pred_fn(test_bags, km=full_km[-n_test:, :n_train])

        if not save_fit:
            self.save_bags = old_save_bags
            self.clear_fit()
        return preds

    ############################################################################
    ### Parameter tuning
    def _param_grid_dict(self):
        return {
            'sigma': np.sort(self.sigma_vals),
        }

    def _set_tuning(self, d):
        for k, v in iteritems(d):
            setattr(self, k + '_', v)

    def _tuned_params(self):
        return dict(
            (name[:-1], getattr(self, name))
            for name in dir(self)
            if name.endswith('_') and not name.startswith('_')
                and name not in ('svm_', 'test_transformer_', 'tune_evals_')
        )

    def _svm_params(self, tuning=False):
        if tuning:
            d = {
                'cache_size': self.tuning_cache_size,
                'tol': self.tuning_svm_tol,
                'max_iter': self.tuning_svm_max_iter,
            }
        else:
            d = {
                'cache_size': self.cache_size,
                'tol': self.svm_tol,
                'max_iter': self.svm_max_iter,
            }
        d['kernel'] = 'precomputed'
        d['shrinking'] = self.svm_shrinking
        return d

    def _tune_params(self, divs, labels, sample_weight=None):
        # TODO: support tuning based on the same inductive technique we'll
        #       predict with: github.com/dougalsutherland/py-sdm/issues/21
        #       (this will probably make tuning a decent bit more expensive)

        # check input shapes
        num_folds = self.tuning_folds
        num_bags = divs.shape[0]
        if not (divs.ndim == 2 and divs.shape[1] == num_bags):
            msg = "divs is {}, should be ({1},{1})".format(divs.shape, num_bags)
            raise ValueError("divs is wrong shape")
        if labels.shape != (num_bags,):
            msg = "labels is {}, should be ({},)".format(labels.shape, num_bags)
            raise ValueError(msg)
        if sample_weight is not None and sample_weight.shape != (num_bags,):
            msg = "sample_weight is {}, should be ({},)".format(
                    sample_weight.shape, num_bags)
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
            self._set_tuning(next(iter(param_grid)))
            self.tune_evals_ = (None, [], {})
            return

        # get kernel matrices for the sigma vals we're trying
        # TODO: could be more careful about making copies here
        sigma_kms = {}
        self.status_fn('Projecting...')
        for sigma in param_d['sigma']:
            #status_fn('Projecting: sigma = {}'.format(sigma))
            km = make_km(divs, sigma, method=self.km_method)
            sigma_kms[sigma] = ForkedData(km)

        labels_d = ForkedData(labels)
        sample_weight_d = ForkedData(sample_weight)

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

        # function that gets loss for a given set of params
        try_params = partial(_try_params, self.__class__,
                             sigma_kms=sigma_kms, labels=labels_d, folds=folds,
                             svm_params=self._svm_params(tuning=True),
                             sample_weight=sample_weight_d)

        # actually do it
        problems = Counter()
        with get_pool(self.n_proc) as pool:
            for ps, val, status in pool.imap_unordered(try_params, param_grid):
                idx = tuple(param_d[k].searchsorted(ps[k]) for k in param_names)
                scores[idx] = val
                if status:
                    problems[status] += 1
                if self.progressbar:
                    tick_pbar()

        if self.progressbar:
            pbar.finish()

        warnings.filters.remove(ignore_conv)
        if problems:
            for msg, count in iteritems(problems):
                if msg == 'convergence warning':
                    continue
                self.status_fn("{} SVMs got error: {}".format(count, msg))

            if problems['convergence warning']:
                msg = '{} SVMs terminated early, after {:,} steps'
                self.status_fn(msg.format(problems['convergence warning'],
                                          self.tuning_svm_max_iter))
            else:
                msg = "All other SVMs finished within {:,} steps"
                self.status_fn(msg.format(self.tuning_svm_max_iter))
        else:
            self.status_fn('All SVMs finished within {:,} steps'.format(
                self.tuning_svm_max_iter))

        # figure out which ones were best
        assert not np.any(np.isnan(scores))
        fold_idx_idx = param_names.index('fold_idx')
        nonfold_param_names = (
                param_names[:fold_idx_idx] + param_names[fold_idx_idx+1:])

        cv_means = scores.mean(axis=fold_idx_idx)
        best_elts = cv_means == cv_means.min()
        best_indices = random.choice(np.transpose(best_elts.nonzero()))
        assert len(nonfold_param_names) == len(best_indices)
        the_params = dict(
                (name, param_d[name][idx])
                for name, idx in izip(nonfold_param_names, best_indices))
        self.status_fn("Chose params {}; tuning loss {}".format(
            ', '.join('{}={}'.format(k, v) for k, v in iteritems(the_params)),
            cv_means[tuple(best_indices)]))
        self._set_tuning(the_params)

        # TODO: save this in a nice format
        del param_d['fold_idx']
        self.tune_evals_ = (cv_means, nonfold_param_names, param_d)

    ############################################################################
    ### Cross-validation helper
    def crossvalidate(self, bags, labels, project_all=True,
                      num_folds=10, stratified_cv=False, folds=None,
                      ret_fold_info=False, ret_tune_info=False,
                      divs=None, divs_cache=None):
        # TODO: document crossvalidate()
        # TODO: nicer interface for ret_tune_info
        status = self.status_fn

        labels = np.squeeze(labels)
        if labels.ndim != 1:
            raise TypeError("train_labels should be 1d")
        num_bags = labels.shape[0]

        if self.classifier:
            # TODO: be nicer about this (support rounding)
            if not is_categorical_type(labels) or labels.min() < 0:
                raise TypeError("train_labels should be nonnegative integers")
        else:
            if not np.all(np.isfinite(labels)):
                raise TypeError("train_labels should be finite")

        if divs is None:
            if not isinstance(bags, Features):
                bags = Features(bags)
            status('Getting divergences...')
            divs = get_divs_cache(bags,
                    div_func=self.div_func, K=self.K,
                    cache_filename=divs_cache, n_proc=self.n_proc,
                    min_dist=self.min_dist,
                    status_fn=self._status_fn, progressbar=self.progressbar)
        else:
            if divs.shape != (num_bags, num_bags):
                raise ValueError("divs should be num_bags x num_bags")

        if self.classifier:
            preds = -np.ones(num_bags, dtype=int)
        else:
            preds = np.empty(num_bags)
            preds.fill(np.nan)

        if folds is None:
            if stratified_cv:
                folds = StratifiedKFold(labels, n_folds=num_folds)
            else:
                folds = KFold(n=num_bags, n_folds=num_folds, shuffle=True)
        else:
            num_folds = len(folds)

        old_save_bags = self.save_bags
        self.save_bags = False  # avoid keeping copies around

        params = []
        tune_info = []
        for i, (train, test) in enumerate(folds, 1):
            status('')
            status('Starting fold {} / {}'.format(i, num_folds))

            both = np.hstack((train, test))

            if self.classifier:
                status('Train distribution: {}'.format(
                        dict(Counter(labels[train]))))
                status('Test distribution: {}'.format(
                        dict(Counter(labels[test]))))

            if project_all:
                preds[test] = self.transduct(
                        None, labels[train], None,
                        divs=divs[np.ix_(both, both)], save_fit=True)
            else:
                self.fit(None, labels[train], divs=divs[np.ix_(train, train)])
                pred_divs = (divs[np.ix_(test, train)] +
                             divs[np.ix_(train, test)].T) / 2
                preds[test] = self.predict(None, divs=pred_divs)

            score = self.eval_score(labels[test], preds[test])
            status('Fold {score_name}: {score:{score_fmt}}'.format(score=score,
                        score_name=self.score_name, score_fmt=self.score_fmt))

            params.append(self._tuned_params())
            tune_info.append(self.tune_evals_)
            self.clear_fit()

        self.save_bags = old_save_bags

        score = self.eval_score(labels, preds)
        ret = (score, preds)
        if ret_fold_info:
            keys = reduce(set.union, (set(iterkeys(x)) for x in params))
            params_a = np.array(
                [tuple(p.get(k, np.nan) for k in keys) for p in params],
                dtype=[(k, np.float) for k in keys])
            # TODO: allow for more general dtypes?

            ret += (folds, params_a)
            if ret_tune_info:
                ret += (tune_info,)  # TODO: nicer format for tune_info
        return ret
        # TODO: nicer output arguments in general (namedtuple)


class BaseSDMClassifier(BaseSDM):
    classifier = True
    tuning_loss = staticmethod(zero_one_loss)
    eval_score = staticmethod(accuracy_score)
    score_name = 'accuracy'
    score_fmt = '.1%'

    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 probability=DEFAULT_SVM_PROBABILITY,
                 status_fn=None, progressbar=None,
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        super(BaseSDMClassifier, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_max_iter=svm_max_iter, tuning_svm_max_iter=tuning_svm_max_iter,
            svm_shrinking=svm_shrinking,
            status_fn=status_fn, progressbar=progressbar,
            min_dist=min_dist,
            symmetrize_divs=symmetrize_divs,
            km_method=km_method,
            transform_test=transform_test,
            save_bags=save_bags)
        self.probability = probability

    def _check_proba(self):
        if not self.probability:
            raise NotImplementedError(
                "probability estimates must be enabled to use this method")
        elif getattr(self, 'svm_', None) is not None:
            pass  # hasn't been fit; _prediction_km() will complain right away
        elif not self.svm_.probability:
            msg = ("Although probability estimates are enabled, it looks like "
                   "they were turned on after fitting. That won't work.")
            raise ValueError(msg)

    def predict_proba(self, data, divs=None, km=None):
        self._check_proba()
        km = self._prediction_km(data, divs=divs, km=km)
        return self.svm_.predict_proba(km)

    def predict_log_proba(self, data, divs=None, km=None):
        self._check_proba()
        km = self._prediction_km(data, divs=divs, km=km)
        return self.svm_.predict_log_proba(km)

    def fit(self, X, y, sample_weight=None, divs=None, divs_cache=None,
            ret_km=False):
        return super(BaseSDMClassifier, self).fit(
            X, y, sample_weight=sample_weight, divs=divs, divs_cache=divs_cache,
            ret_km=ret_km)
    fit.__doc__ = BaseSDM._fit_docstr.format(y_doc="""
        y: a vector of nonnegative integer class labels.
            -1 corresponds to data that should be used semi-supervised, i.e. its
            divergences are calculated and used to project the Gram matrix, but
            are not used in training the SVM.
    """)


class SDC(BaseSDMClassifier):
    svm_class = svm.SVC

    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
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
                 probability=DEFAULT_SVM_PROBABILITY,
                 status_fn=None, progressbar=None,
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        super(SDC, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_max_iter=svm_max_iter, tuning_svm_max_iter=tuning_svm_max_iter,
            svm_shrinking=svm_shrinking,
            probability=probability,
            status_fn=status_fn, progressbar=progressbar,
            min_dist=min_dist,
            symmetrize_divs=symmetrize_divs,
            km_method=km_method,
            transform_test=transform_test,
            save_bags=save_bags)
        self.C_vals = C_vals
        self.weight_classes = weight_classes

    def _param_grid_dict(self):
        d = super(SDC, self)._param_grid_dict()
        d['C'] = np.sort(self.C_vals)
        return d

    def _svm_params(self, tuning=False):
        d = super(SDC, self)._svm_params(tuning=tuning)
        d['class_weight'] = 'auto' if self.weight_classes else None
        if not tuning:
            d['C'] = self.C_
        return d


class NuSDC(BaseSDMClassifier):
    svm_class = svm.NuSVC

    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 nu_vals=DEFAULT_NU_VALS,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 probability=DEFAULT_SVM_PROBABILITY,
                 status_fn=None, progressbar=None,
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        super(NuSDC, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_max_iter=svm_max_iter, tuning_svm_max_iter=tuning_svm_max_iter,
            svm_shrinking=svm_shrinking,
            probability=probability,
            status_fn=status_fn, progressbar=progressbar,
            min_dist=min_dist,
            symmetrize_divs=symmetrize_divs,
            km_method=km_method,
            transform_test=transform_test,
            save_bags=save_bags)
        self.nu_vals = nu_vals

    def _param_grid_dict(self):
        d = super(NuSDC, self)._param_grid_dict()
        d['nu'] = np.sort(self.nu_vals)
        return d

    def _svm_params(self, tuning=False):
        d = super(NuSDC, self)._svm_params(tuning=tuning)
        if not tuning:
            d['nu'] = self.nu_
        return d


class BaseSDMRegressor(BaseSDM):
    regressor = True
    tuning_loss = staticmethod(mean_squared_error)
    eval_score = staticmethod(rmse)
    score_name = 'RMSE'
    score_fmt = ''

    def fit(self, X, y, sample_weight=None, divs=None, divs_cache=None,
            ret_km=False):
        return super(BaseSDMRegressor, self).fit(
            X, y, sample_weight=sample_weight, divs=divs, divs_cache=divs_cache,
            ret_km=ret_km)
    fit.__doc__ = BaseSDM._fit_docstr.format(y_doc="""
        y: a vector of real-valued labels.
            nan corresponds to data that should be used semi-supervised, i.e.
            its divergences are calculated and used to project the Gram matrix,
            but are not used in training the SVM.
        """)


class SDR(BaseSDMRegressor):
    svm_class = svm.SVR

    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 C_vals=DEFAULT_C_VALS,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 svr_epsilon_vals=DEFAULT_SVR_EPSILON_VALS,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 status_fn=None, progressbar=None,
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        super(SDR, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_max_iter=svm_max_iter, tuning_svm_max_iter=tuning_svm_max_iter,
            svm_shrinking=svm_shrinking,
            status_fn=status_fn, progressbar=progressbar,
            min_dist=min_dist,
            symmetrize_divs=symmetrize_divs,
            km_method=km_method,
            transform_test=transform_test,
            save_bags=save_bags)
        self.C_vals = C_vals
        self.svr_epsilon_vals = svr_epsilon_vals

    def _param_grid_dict(self):
        d = super(SDR, self)._param_grid_dict()
        d['C'] = np.sort(self.C_vals)
        d['epsilon'] = np.sort(self.svr_epsilon_vals)
        return d

    def _svm_params(self, tuning=False):
        d = super(SDR, self)._svm_params(tuning=tuning)
        if not tuning:
            d['C'] = self.C_
            d['epsilon'] = self.svr_epsilon_
        return d

    def _set_tuning(self, d):
        self.svr_epsilon_ = d.pop('epsilon')
        super(SDR, self)._set_tuning(d)


class NuSDR(BaseSDMRegressor):
    svm_class = svm.NuSVR

    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
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
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        super(NuSDR, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            sigma_vals=sigma_vals, scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_max_iter=svm_max_iter, tuning_svm_max_iter=tuning_svm_max_iter,
            svm_shrinking=svm_shrinking,
            status_fn=status_fn, progressbar=progressbar,
            min_dist=min_dist,
            symmetrize_divs=symmetrize_divs,
            km_method=km_method,
            transform_test=transform_test,
            save_bags=save_bags)
        self.C_vals = C_vals
        self.svr_nu_vals = svr_nu_vals

    def _param_grid_dict(self):
        d = super(NuSDR, self)._param_grid_dict()
        d['C'] = np.sort(self.C_vals)
        d['nu'] = np.sort(self.svr_nu_vals)
        return d

    def _svm_params(self, tuning=False):
        d = super(NuSDR, self)._svm_params(tuning=tuning)
        if not tuning:
            d['C'] = self.C_
            d['nu'] = self.svr_nu_
        return d

    def _set_tuning(self, d):
        self.svr_nu_ = d.pop('nu')
        super(NuSDR, self)._set_tuning(d)


class OneClassSDM(BaseSDM):
    oneclass = True
    svm_class = svm.OneClassSVM
    # leaving tuning_loss, eval_score, score_name, score_fmt unimplemented
    # TODO: add real tuning support for one-class SDMs

    def __init__(self,
                 div_func=DEFAULT_DIV_FUNC,
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 nu=0.5,
                 sigma=1, scale_sigma=True,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 status_fn=None, progressbar=None,
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True):
        super(OneClassSDM, self).__init__(
            div_func=div_func, K=K, tuning_folds=tuning_folds, n_proc=n_proc,
            sigma_vals=np.array([sigma]), scale_sigma=scale_sigma,
            cache_size=cache_size, tuning_cache_size=tuning_cache_size,
            svm_tol=svm_tol, tuning_svm_tol=tuning_svm_tol,
            svm_max_iter=svm_max_iter, tuning_svm_max_iter=tuning_svm_max_iter,
            svm_shrinking=svm_shrinking,
            status_fn=status_fn, progressbar=progressbar,
            min_dist=min_dist,
            symmetrize_divs=symmetrize_divs,
            km_method=km_method,
            transform_test=transform_test,
            save_bags=save_bags)
        self.nu_vals = np.array([nu])

    def _param_grid_dict(self):
        d = super(OneClassSDM, self)._param_grid_dict()
        d['nu'] = np.sort(self.nu_vals)
        return d

    def _svm_params(self, tuning=False):
        d = super(OneClassSDM, self)._svm_params(tuning=tuning)
        if not tuning:
            d['nu'] = self.nu_
        return d

    def fit(self, X, sample_weight=None, divs=None, divs_cache=None,
            ret_km=False):
        if X is not None:
            y = np.zeros(len(X))  # fake labels so superclass doesn't flip out
        elif divs is not None:
            y = np.zeros(divs.shape[0])
        else:
            raise ValueError("need to pass either X or divs to fit")
        return super(OneClassSDM, self).fit(
                X, y, sample_weight=sample_weight, ret_km=ret_km,
                divs=divs, divs_cache=divs_cache)
    fit.__doc__ = BaseSDM._fit_docstr.format(y_doc='')


sdm_for_mode = {
    'SVC': SDC,
    'NuSVC': NuSDC,
    'SVR': SDR,
    'NuSVR': NuSDR,
    'OneClassSVM': OneClassSDM,
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

        algo._add_action(ActionNoYes('transform-test',
            default=DEFAULT_TRANSFORM_TEST,
            help="When operating inductively, transform the test similarities "
                 "to treat them consistently with our transformations on the "
                 "training data" +
                 (" (the default)." if DEFAULT_TRANSFORM_TEST else ".")))

        m = algo.add_mutually_exclusive_group()
        m.set_defaults(km_method=DEFAULT_KM_METHOD)
        m.add_argument('--psd-clip',
            dest='km_method', action='store_const', const='clip',
            help="Make kernel matrices by projecting to the nearest PSD matrix "
                 "in Frobenius norm (clipping negative eigenvalues to 0).")
        m.add_argument('--psd-flip',
            dest='km_method', action='store_const', const='flip',
            help="Make kernel matrices PSD by flipping negative eigenvalues to "
                 " be positive.")
        m.add_argument('--psd-shift',
            dest='km_method', action='store_const', const='shift',
            help="Make kernel matrices PSD by adding to the diagonal until "
                 "it's PSD.")
        m.add_argument('--psd-square',
            dest='km_method', action='store_const', const='shift',
            help="Make kernel matrices PSD by squaring them (S S^T). Amounts "
                 "to using the similarities as features.")

        algo._add_action(ActionNoYes('symmetrize-divs', default=False,
            help="Symmetrize divergence estimates before passing them through "
                 "the RBF kernel, rather than after."))

        m = algo.add_mutually_exclusive_group()
        m.set_defaults(svm_mode='SVC')
        m.add_argument('--svc',
            action='store_const', dest='svm_mode', const='SVC',
            help="Use the standard support vector classifier (default), "
                 "whose parameter is a slack penalty weight C.")
        m.add_argument('--nu-svc',
            action='store_const', dest='svm_mode', const='NuSVC',
            help="Use a nu support vector classifier, whose parameter is "
                 "a lower bound on the fraction of support vectors nu.")
        m.add_argument('--svr',
            action='store_const', dest='svm_mode', const='SVR',
            help="Use a support vector regressor, whose parameters are "
                 "a slack penalty weight C and the width of the regression "
                 "tube epsilon.")
        m.add_argument('--nu-svr',
            action='store_const', dest='svm_mode', const='NuSVR',
            help="Use a nu support vector regressor, whose parameters are "
                 "a slack penalty weight C and a lower bound on the fraction "
                 "of support vectors nu.")

        algo.add_argument('--div-func', '-d', default=DEFAULT_DIV_FUNC,
            type=normalize_div_name,
            help="The divergence function to use " + _def)

        algo.add_argument('-K', type=positive_int, default=DEFAULT_K,
            help="How many nearest neighbors to use " + _def)

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

        algo.add_argument('--sigma-vals', '-S', type=positive_float, nargs='+',
            default=DEFAULT_SIGMA_VALS, metavar='SIGMA',
            help="Values to try for tuning kernel bandwidth sigma " + _def)
        algo._add_action(ActionNoYes('scale-sigma', default=True,
            help="Scale --sigma-vals by the median nonzero divergence; "
                 "does by default."))
        algo.add_argument('--c-vals', '-C', type=positive_float, nargs='+',
            default=DEFAULT_C_VALS, metavar='C',
            help="Values to try for tuning SVM regularization strength " + _def)
        algo.add_argument('--nu-vals', type=portion, nargs='+',
            default=DEFAULT_NU_VALS, metavar='NU',
            help="Values to try for tuning the nu of NuSVC, a lower bound on "
                 "the fraction of support vectors " + _def)
        algo.add_argument('--svr-epsilon-vals', type=positive_float, nargs='+',
            default=DEFAULT_SVR_EPSILON_VALS, metavar='EPSILON',
            help="Values to try for tuning the epsilon of SVR, the amount "
                 "within which we count a regression as good enough " + _def)
        algo.add_argument('--svr-nu-vals', type=portion, nargs='+',
            default=DEFAULT_SVR_NU_VALS, metavar='NU',
            help="Values to try for tuning the nu of NuSVR, a lower bound on "
                 "the fraction of support vectors " + _def)

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
        help="Name of the output file; defaults to input_file.sdm_cv.(mat|npz).")
    io.add_argument('--output-format', choices=['npz', 'mat'], default='npz',
        help="Format of the output file " + _def)

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
            'predict': '.sdm_preds.',
            'cv': '.sdm_cv.',
        }
        formats = {
            'mat': 'mat',
            'npz': 'npz',
        }
        args.output_file = args.input_file + suffixes[args.subcommand] + \
                           formats[args.output_format]

    args.sigma_vals = np.sort(args.sigma_vals)
    args.c_vals = np.sort(args.c_vals)
    args.nu_vals = np.sort(args.nu_vals)
    args.svr_nu_vals = np.sort(args.svr_nu_vals)
    args.svr_epsilon_vals = np.sort(args.svr_epsilon_vals)

    return args


def opts_dict(args):
    d = {
        'div_func': args.div_func,
        'K': args.K,
        'tuning_folds': args.tuning_folds,
        'n_proc': args.n_proc,
        'sigma_vals': args.sigma_vals, 'scale_sigma': args.scale_sigma,
        'cache_size': args.cache_size,
        'tuning_cache_size': args.tuning_cache_size,
        'svm_tol': args.svm_tol,
        'tuning_svm_tol': args.tuning_svm_tol,
        'svm_max_iter': args.svm_max_iter,
        'tuning_svm_max_iter': args.tuning_svm_max_iter,
        'svm_shrinking': args.svm_shrinking,
        'min_dist': args.min_dist,
        'symmetrize_divs': args.symmetrize_divs,
        'km_method': args.km_method,
        'transform_test': args.transform_test,
    }
    # TODO: switch to subparsers based on svm type to only accept the right args
    if args.svm_mode == 'SVC':
        d['C_vals'] = args.c_vals
        d['weight_classes'] = args.weight_classes
    elif args.svm_mode == 'NuSVC':
        d['nu_vals'] = args.nu_vals
    elif args.svm_mode == 'SVR':
        d['C_vals'] = args.c_vals
        d['svr_epsilon_vals'] = args.svr_epsilon_vals
    elif args.svm_mode == 'NuSVR':
        d['C_vals'] = args.c_vals
        d['svr_nu_vals'] = args.svr_nu_vals
    else:
        raise ValueError("can't handle svm_mode {!r}".format(args.svm_mode))
    return d


def do_predict(args):
    import h5py
    status_fn = get_status_fn(True)

    status_fn('Reading inputs...')
    with h5py.File(args.input_file, 'r') as f:
        train_bags = read_cell_array(f, f[args.train_bags_name])
        train_labels = f[args.train_labels_name][...]
        test_bags = read_cell_array(f, f[args.test_bags_name])

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
    import h5py
    status_fn = get_status_fn(True)

    classifier = args.svm_mode in ('SVC', 'NuSVC')

    status_fn('Reading inputs...')
    if args.input_format == 'matlab':
        with h5py.File(args.input_file, 'r') as f:
            bags = read_cell_array(f, f[args.bags_name])
            try:
                cats = np.squeeze(f['cats'][...])
            except KeyError:
                cats = None

            feats = Features(bags, categories=cats)

            if args.labels_name:
                labels = np.squeeze(f[args.labels_name][...])
            elif cats is not None:
                labels = cats
            else:
                raise ValueError("must provide a label name")
    else:
        assert args.input_format == 'python'

        if os.path.isdir(args.input_file):
            feats = Features.load_from_perbag(args.input_file)
        else:
            feats = Features.load_from_hdf5(args.input_file)

        if args.labels_name:
            labels = feats[args.labels_name]
        elif args.labels_name is None and classifier:
            labels = feats.categories
        else:
            raise ValueError("must provide a label name when regressing")

    label_class_names = None
    if classifier and not is_categorical_type(labels):
        if labels.dtype.kind == 'f' and np.all(labels == np.round(labels)):
            labels = labels.astype(int)
        else:
            label_names = labels
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(label_names)
            label_class_names = label_encoder.classes_

    if classifier:
        label_str = '{} classes'.format(len(set(labels)))
    else:
        label_str = 'labels from {:.2} to {:.2}'.format(
                np.min(labels), np.max(labels))
    status_fn('Loaded {} with {}.'.format(feats, label_str))

    clf = sdm_for_mode[args.svm_mode](status_fn=True, **opts_dict(args))
    score, preds, folds, params = clf.crossvalidate(feats, labels,
        num_folds=args.cv_folds, stratified_cv=args.stratified_cv,
        project_all=args.mode == 'transduct',
        ret_fold_info=True,
        divs_cache=args.div_cache_file)

    status_fn('')
    status_fn('{score_name}: {score:{score_fmt}}'.format(score=score,
                score_name=clf.score_name[:1].upper() + clf.score_name[1:],
                score_fmt=clf.score_fmt))

    out = {
        'div_func': args.div_func,
        'K': args.K,
        'scale_sigma': args.scale_sigma,
        'preds': preds,
        'labels': labels,
        'svm_mode': args.svm_mode,
        'folds': list(folds),
        'params': params,
    }
    for k, v in iteritems(clf._param_grid_dict()):
        out[k + '_vals'] = v
    out[clf.score_name] = score
    if label_class_names is not None:
        out['label_names'] = label_class_names
    status_fn('Saving output to {}'.format(args.output_file))
    if args.output_format == 'mat':
        scipy.io.savemat(args.output_file, out, oned_as='column')
    else:
        np.savez(args.output_file, **out)


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
