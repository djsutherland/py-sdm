#!/usr/bin/env python
from __future__ import division, print_function

from functools import partial
import os
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .features import Features, DEFAULT_VARFRAC
from .utils import positive_int, portion, nonnegative_float, confirm_outfile

_do_nothing = lambda *a, **k: None


################################################################################
### Handling blank descriptors

DEFAULT_BLANK_HANDLER = 'fill'
DEFAULT_BLANK_THRESH = 1000

class FillBlanks(BaseEstimator, TransformerMixin):
    '''
    Fills in any almost-blank SIFT descriptors with a random one.

    copy: if False, change the input in-place. By default, operates on a copy.

    blank_thresh: the threshold for considering a descriptor to be blank.
        Any descriptors whose sum is less than this threshold are replaced.
    '''

    def __init__(self, copy=True, blank_thresh=DEFAULT_BLANK_THRESH):
        self.copy = copy
        self.blank_thresh = blank_thresh

    def fit(self, X, y=None):
        "Does nothing, since this transform doesn't require fitting."
        pass

    def transform(self, X, y=None, copy=None):
        copy = copy if copy is not None else self.copy
        n_f, dim = X.shape
        if copy:
            X = X.copy()

        mag = 3 * (2 * self.blank_thresh / dim)

        blank_idx = np.sum(X, axis=1) < self.blank_thresh
        X[blank_idx, :] = mag * np.random.rand(blank_idx.sum(), dim)
        return X


class ZeroBlanks(BaseEstimator, TransformerMixin):
    '''
    Zeroes out any almost-blank SIFT descriptors.

    copy: if False, change the input in-place. By default, operates on a copy.

    blank_thresh: the threshold for considering a descriptor to be blank.
        Any descriptors whose sum is less than this threshold are replaced.
    '''

    def __init__(self, copy=True, blank_thresh=DEFAULT_BLANK_THRESH):
        self.copy = copy
        self.blank_thresh = blank_thresh

    def fit(self, X, y=None):
        "Does nothing, since this transform doesn't require fitting."
        pass

    def transform(self, X, y=None, copy=None):
        copy = copy if copy is not None else self.copy
        if copy:
            X = X.copy()
        X[np.sum(X, axis=1) < self.blank_thresh, :] = 0
        return X


BLANK_HANDLERS = {
    'fill': FillBlanks,
    'zero': ZeroBlanks,
    'drop': None,
}

def handle_blanks(features, blank_thresh=DEFAULT_BLANK_THRESH,
                            blank_handler=DEFAULT_BLANK_HANDLER,
                            inplace=False):
    '''Handles any SIFT descriptors that are blank, or nearly blank.'''

    if blank_handler not in BLANK_HANDLERS:
        msg = "unknown blank handler {!r}, expected one of {}".format(
            blank_handler, ", ".join(map(repr, BLANK_HANDLERS)))
        raise ValueError(msg)

    if blank_handler == 'drop':
        # TODO handle this more efficiently
        feats = [
            f[np.sum(f, axis=1) >= blank_thresh, :] for f in features.features
        ]

        args = dict((k, features.data[k]) for k in f._extra_names)
        args['categories'] = features.categories
        args['names'] = features.names

        if inplace:
            features.__init__(feats, **args)
            return
        else:
            return Features(feats, **args)

    handler = BLANK_HANDLERS[blank_handler](blank_thresh=blank_thresh)
    r = features._apply_transform(handler, fit_first=True, inplace=inplace)

    if not inplace:
        return r


################################################################################
### Add spatial information

def add_spatial_info(features, add_x=True, add_y=True, inplace=False,
                     dtype=None):
    '''
    Adds spatial information to image features (which should contain a frames
    attribute in the format created by extract_image_features).

    Adds a feature for x (if add_x) and y (if add_y), which are relative (x, y)
    locations within the image of the feature between 0 and 1 (inclusive).

    Returns a new Features object with these additional features, or modifies
    features and returns None if inplace is True.

    If dtype is not None, the resulting array will have that dtype. Otherwise,
    it will maintain features.dtype if it's a float type, or float32 if not.
    '''
    if not add_x and not add_y:
        return None if inplace else features

    indices = []
    if add_x:
        indices.append(0)
    if add_y:
        indices.append(1)

    if dtype is None:
        dtype = features.dtype
        if dtype.kind != 'f':
            dtype = np.float32

    spatial = np.asarray(np.vstack(features.frames)[:, indices], dtype=dtype)
    spatial /= spatial.max(axis=0)

    new_feats = np.hstack((features._features, spatial))
    if inplace:
        features._features = new_feats
        features._refresh_features()
    else:
        return Features(
            new_feats, n_pts=features._n_pts,
            categories=features.categories, names=features.names,
            **dict((k, features.data[k]) for k in features._extra_names))


################################################################################
### Wrapper for general processing

def process_image_features(features, verbose=False, inplace=False,
        blank_thresh=DEFAULT_BLANK_THRESH, blank_handler=DEFAULT_BLANK_HANDLER,
        do_pca=True, pca=None,
            pca_k=None, pca_varfrac=DEFAULT_VARFRAC, pca_random=False,
            pca_whiten=False,
        add_x=True, add_y=True,
        standardize_feats=True, scaler=None,
        ret_pca=False, ret_scaler=False):
    '''
    Does the full image processing stack:
        - blank handling with handle_blanks()
        - dimensionality reduction with features.pca()
        - adds spatial information with add_spatial_info()
        - standardizes the features with features.standardize()
    '''
    # TODO: use sklearn.Pipeline instead?
    pr = partial(print, file=sys.stderr) if verbose else _do_nothing

    if blank_handler not in (None, "none"):
        pr("Handling blanks...")
        ret = handle_blanks(features, blank_thresh=blank_thresh,
                            blank_handler=blank_handler, inplace=inplace)
        if not inplace:
            features = ret

    if do_pca:
        pr("Running PCA...")
        old_dim = features.dim

        ret = features.pca(
            pca=pca, ret_pca=True, k=pca_k, varfrac=pca_varfrac,
            randomize=pca_random, whiten=pca_whiten, inplace=inplace)
        if inplace:
            pca = ret
        else:
            features, pca = ret

        new_dim = features.dim
        pr("Reduced dimensionality from {} to {}.".format(old_dim, new_dim))

    if add_x or add_y:
        pr("Adding spatial info...")
        ret = add_spatial_info(features, add_x, add_y, inplace=inplace)
        if not inplace:
            features = ret

    if normalize_feats:
        pr("Standardizing features...")
        ret = features.standardize(scaler=scaler, ret_scaler=True,
                                   inplace=inplace)
        if inplace:
            scaler = ret
        else:
            features, scaler = ret

    if not ret_pca and not ret_scaler:
        return features
    ret = [features]
    if ret_pca:
        ret.append(pca)
    if ret_scaler:
        ret.append(scaler)
    return ret


def parse_args(args=None):
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

    _def = "; default %(default)s."
    parser = argparse.ArgumentParser(
        description="Processes raw PHOW features before running the SDM.")

    blanks = parser.add_argument_group("Blank handling")
    blanks.add_argument('--blank-threshold', dest='blank_thresh',
        default=DEFAULT_BLANK_THRESH, type=nonnegative_float,
        help="Consider descriptors with sum < BLANK_THRESH to be blank" + _def)
    blanks.add_argument('--blank-handler',
        choices=list(BLANK_HANDLERS) + ['none'], default=DEFAULT_BLANK_HANDLER,
        help="What to do with blanks" + _def)

    pca = parser.add_argument_group("PCA options")
    pca._add_action(ActionNoYes('do-pca', 'no-pca', default=True,
        help="Whether to run PCA; does by default."))
    dim = pca.add_mutually_exclusive_group()
    dim.add_argument('--pca-k', type=positive_int, default=None,
        help="An explicit dimensionality to reduce the features to.")
    dim.add_argument('--pca-varfrac', default=DEFAULT_VARFRAC,
        type=portion, metavar='FRAC',
        help="The fraction of variance to maintain in the PCA" + _def)
    pca._add_action(ActionNoYes('pca-random', default=False,
        help="Whether to use a randomized PCA implementation; default don't."))
    pca._add_action(ActionNoYes('pca-whiten', default=False,
        help="Whether to do whitening in the PCA, removing linear correlations "
             "between dimensions; default don't."))

    spa = parser.add_argument_group('Spatial information')
    spa._add_action(ActionNoYes('add-x', default=True,
        help="Append normalized x coord of patches; does by default."))
    spa._add_action(ActionNoYes('add-y', default=True,
        help="Append normalized y coord of patches; does by default."))

    std = parser.add_argument_group('Standardization')
    std._add_action(ActionNoYes('normalize-feats', default=True,
        help="Normalize features to mean 0, variance 1 at the end (default)."))

    parser._add_action(ActionNoYes('verbose', 'quiet', default=True,
        help="Print out info along the way (the default)."))

    parser.add_argument('load_file',
        help="Load features from this file (output of extract_features).")
    parser.add_argument('save_file', help="Save into this file.")

    args = parser.parse_args(args)
    load = args.load_file
    save = args.save_file
    del args.load_file, args.save_file
    return args, load, save


def main():
    args, load_file, save_file = parse_args()
    confirm_outfile(save_file)

    pr = partial(print, file=sys.stderr) if args.verbose else _do_nothing

    pr("Loading features from '{}'...".format(load_file))
    kwargs = {'load_attrs': True, 'features_dtype': np.float32}
    if os.path.isdir(load_file):
        orig, attrs = Features.load_from_perbag(load_file, **kwargs)
    else:
        orig, attrs = Features.load_from_hdf5(load_file, **kwargs)

    new, pca, scaler = process_image_features(
        orig, ret_pca=True, ret_scaler=True, **vars(args))

    if pca is not None:
        attrs['pca_mean'] = pca.mean_
        attrs['pca_components'] = pca.components_
    if scaler is not None:
        attrs['scaler_mean'] = scaler.mean_
        attrs['scaler_std'] = scaler.std_

    pr("Saving features to '{}'...".format(save_file))
    new.save_as_hdf5(save_file, process_args=repr(vars(args)), **attrs)


if __name__ == '__main__':
    main()
