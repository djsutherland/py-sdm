#!/usr/bin/env python
from __future__ import division, print_function

from functools import partial
import sys

import numpy as np

from utils import izip, positive_int, portion, nonnegative_float, strict_map
from extract_features import (Features, features_attrs,
                              read_features, save_features, confirm_outfile)

# NOTE: all the references to "features" in this file mean a variable that is
#       like Features.features; "features_tup" means an instance of Features

_do_nothing = lambda *a, **k: None

BLANK_HANDLERS = frozenset(['fill', 'drop', 'zero'])
DEFAULT_BLANK_HANDLER = 'fill'
DEFAULT_BLANK_THRESH = 1000
def handle_blanks(features, blank_thresh=DEFAULT_BLANK_THRESH,
                            blank_handler=DEFAULT_BLANK_HANDLER):
    '''Handles any descriptors that are blank, or nearly blank.'''
    if blank_handler not in BLANK_HANDLERS:
        msg = "unknown blank handler {!r}, expected one of {}".format(
            blank_handler, ", ".join(map(repr, BLANK_HANDLERS)))
        raise ValueError(msg)

    feats = []
    for feat in features:
        n_f, dim = feat.shape
        blank_idx = np.sum(feat, axis=1) < blank_thresh

        if blank_handler == 'fill':
            mag = 3 * (2 * blank_thresh / dim)
            feat = feat.copy()
            feat[blank_idx, :] = mag * np.random.rand(blank_idx.sum(), dim)
        elif blank_handler == 'drop':
            feat = feat[np.logical_not(blank_idx), :]
        elif blank_handler == 'zero':
            feat = feat.copy()
            feat[blank_idx, :] = 0

        feats.append(feat)
    return feats


DEFAULT_VARFRAC = 0.7
def pca_features(features, pca=None, k=None, varfrac=DEFAULT_VARFRAC,
                 randomize=False, dtype=None, ret_pca=False,
                 _stacked=None):
    '''
    PCAs a set of features.

    feats: a sequence of row-major feature matrices.

    You can either pass a "pca" argument with .fit() and .transform() methods,
    or some of the following to use one of scikit-learn's options:

        k: a dimensionality to reduce to. Default: use varfrac instead.

        varfrac: the fraction of variance to preserve. Overridden by k.
            Default: 0.7. Can't be used for randomized or sparse PCA.

        randomize: use a randomized PCA implementation. Default: no.

        dtype: the dtype of the feature matrix to use.

    ret_pca: return the PCA object along with transformed inputs. Default: no.
    '''
    # figure out what PCA instance we should use
    if pca is None:
        from sklearn.decomposition import PCA, RandomizedPCA
        if k is None:
            if randomize:
                raise ValueError("can't randomize without a specific k")
            pca = PCA(varfrac, copy=False)
        else:
            pca = (RandomizedPCA if randomize else PCA)(k, copy=False)
    # copy=False is okay, because the arg to fit() is just the stacked copy

    pca.fit(np.vstack(features))
    transformed = strict_map(pca.transform, features)
    return (transformed, pca) if ret_pca else transformed


def add_spatial_info(features, frames, add_x=True, add_y=True):
    if not add_x and not add_y:
        return features

    indices = []
    if add_x:
        indices.append(0)
    if add_y:
        indices.append(1)

    ret = []
    for feat, frame in izip(features, frames):
        spatial = frame[:, indices].astype(feat.dtype)
        spatial /= spatial.max(axis=0)
        ret.append(np.hstack((feat, spatial)))
    return ret


def normalize(features):
    from sklearn.preprocessing import StandardScaler
    # TODO: get mean and variance without explicitly stacking...
    scaler = StandardScaler(copy=False)
    scaler.fit(np.vstack(features))
    return strict_map(scaler.transform, features)


def process_features(features_tup, verbose=False,
        blank_thresh=DEFAULT_BLANK_THRESH, blank_handler=DEFAULT_BLANK_HANDLER,
        do_pca=True, pca_k=None, pca_varfrac=DEFAULT_VARFRAC, pca_random=False,
        add_x=True, add_y=True,
        normalize_feats=True):
    pr = partial(print, file=sys.stderr) if verbose else _do_nothing

    features = features_tup.features

    if blank_handler is not None:
        pr("Handling blanks...")
        features = handle_blanks(features,
            blank_thresh=blank_thresh, blank_handler=blank_handler)

    if do_pca:
        pr("Running PCA...")
        old_dim = features[0].shape[1]
        features = pca_features(features,
            k=pca_k, varfrac=pca_varfrac, randomize=pca_random)
        new_dim = features[0].shape[1]
        pr("Reduced dimensionality from {} to {}.".format(old_dim, new_dim))

    if add_x or add_y:
        pr("Adding spatial info...")
        features = add_spatial_info(features, features_tup.frames, add_x, add_y)

    if normalize_feats:
        pr("Normalizing features to mean 0, variance 1...")
        features = normalize(features)

    return Features(features=features,
        **dict((k, getattr(features_tup, k))
               for k in features_attrs if k != 'features'))


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
    orig, orig_attrs = read_features(load_file, load_attrs=True,
                                     features_dtype=np.float32)

    new = process_features(orig, **vars(args))

    pr("Saving features to '{}'...".format(save_file))
    save_features(save_file, new, process_args=repr(vars(args)), **orig_attrs)


if __name__ == '__main__':
    main()
