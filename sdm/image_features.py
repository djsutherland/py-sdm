from collections import namedtuple
import os

import numpy as np

from .utils import izip, iterkeys, iteritems


features_attrs = ['categories', 'names', 'frames', 'features', 'extras']
Features = namedtuple('Features', features_attrs)


################################################################################
### Stuff relating to hdf5 features files

def save_features(filename, features, **attrs):
    '''
    Saves a Features namedtuple into an HDF5 file.

    Also saves any keyword args as a dateset under '/meta'.

    Each bag is saved as "features" and "frames" in /category/filename;
    any "extras" get added there as a (probably scalar) dataset named by the
    extra's name.
    '''
    import h5py
    with h5py.File(filename) as f:
        for category, name, frames, descrs, extra in izip(*features):
            g = f.require_group(category).create_group(name)
            g['frames'] = frames
            g['features'] = descrs
            if extra:
                for k, v in iteritems(extra):
                    assert k not in ['frames', 'features']
                    g[k] = v

        meta = f.require_group('_meta')
        for k, v in iteritems(attrs):
            meta[k] = v


def read_features(filename, load_attrs=False, features_dtype=None,
                  cats=None, pairs=None, subsample_fn=None,
                  names_only=False):
    '''
    Reads a Features namedtuple from an h5py file created by save_features().

    If load_attrs, also returns a dictionary of meta values loaded from
    root attributes, '/_meta' attributes, '/_meta' datasets.

    features_dtype specifies the datatype to load features as.

    If cats is passed, only load those with a category in cats (as checked
    by the `in` operator, aka the __contains__ special method).

    If pairs is passed, tuples of (category, name) are checked with the `in`
    operator. If cats is also passed, that check applies first.

    subsample_fn is applied to a list of (category, name) pairs, and returns
    another list of that format. functool.partial(random.sample, k=100) can
    be used to subsample 100 bags unifornmly at random, for example.

    If names_only is passed, the list of (category, name) pairs is returned
    without having loaded any data. load_attrs is also ignored.
    '''
    import h5py
    ret = Features(*[[] for _ in features_attrs])

    with h5py.File(filename, 'r') as f:
        bag_names = []
        for cat, cat_g in iteritems(f):
            if cats is None or cat in cats:
                for fname in iterkeys(cat_g):
                    if pairs is None or (cat, fname) in pairs:
                        bag_names.append((cat, fname))

        if subsample_fn is not None:
            bag_names = subsample_fn(bag_names)

        if names_only:
            return bag_names

        for cat, fname in bag_names:
            if cat == '_meta':
                continue

            ret.categories.append(cat)
            ret.names.append(fname)
            extra = {}
            frames = None
            feats = None
            for k, v in iteritems(f[cat][fname]):
                if k == 'frames':
                    frames = v[()]
                elif k == 'features':
                    if features_dtype is not None:
                        feats = np.asarray(v, dtype=features_dtype)
                    else:
                        feats = v[()]
                else:
                    extra[k] = v[()]
            ret.features.append(feats)
            ret.frames.append(frames)
            ret.extras.append(extra)

        if load_attrs:
            attrs = {}
            if '_meta' in f:
                for k, v in iteritems(f['_meta']):
                    attrs[k] = v[()]
                for k, v in iteritems(f['_meta'].attrs):
                    if k not in attrs:
                        attrs[k] = v
            for k, v in iteritems(f.attrs):
                if k not in attrs:
                    attrs[k] = v
            return ret, attrs

        return ret


################################################################################
### Stuff relating to per-image npz feature files

def save_features_perimage(path, features, **attrs):
    '''
    Save a Features namedtuple into per-image npz files.
    '''
    import pickle
    with open(os.path.join(path, 'attrs.pkl'), 'wb') as f:
        pickle.dump(attrs, f)

    for cat, name, frames, features, extras in zip(*features):
        dirpath = os.path.join(path, cat)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        np.savez(os.path.join(dirpath, name + '.npz'),
            frames=frames, features=features, **extras)


def read_features_perimage(path, load_attrs=False, features_dtype=None,
                           cats=None, pairs=None, subsample_fn=None,
                           names_only=False):
    '''
    Reads a Features namedtuple from a directory of npz files created
    by save_features_perimage().

    If load_attrs, also returns a dictionary of meta values loaded from the
    'attrs.pkl' file, if it exists.

    features_dtype specifies the datatype to load features as.

    If cats is passed, only load those with a category in cats (as checked
    by the `in` operator, aka the __contains__ special method).

    If pairs is passed, tuples of (category, name) are checked with the `in`
    operator. If cats is also passed, that check applies first.

    subsample_fn is applied to a list of (category, name) pairs, and returns
    another list of that format. functool.partial(random.sample, k=100) can
    be used to subsample 100 bags unifornmly at random, for example.

    If names_only is passed, the list of (category, name) pairs is returned
    without having loaded any data. load_attrs is also ignored.

    '''
    from glob import glob

    ret = Features(*[[] for _ in features_attrs])

    bag_names = []
    for cat in os.listdir(path):
        dirpath = os.path.join(path, cat)
        if os.path.isdir(dirpath) and (cats is None or cat in cats):
            for npz_fname in glob(os.path.join(dirpath, '*.npz')):
                fname = npz_fname[len(dirpath) + 1:-len('.npz')]
                if pairs is None or (cat, fname) in pairs:
                    bag_names.append((cat, fname))

    if subsample_fn is not None:
        bag_names = subsample_fn(bag_names)

    if names_only:
        return bag_names

    for cat, fname in bag_names:
        ret.categories.append(cat)
        ret.names.append(fname)

        data = np.load(os.path.join(path, cat, fname + '.npz'))
        feats = None
        frames = None
        extra = {}
        for k, v in iteritems(data):
            if k == 'frames':
                frames = v[()]
            elif k == 'features':
                if features_dtype is not None:
                    feats = np.asarray(v, dtype=features_dtype)
                else:
                    feats = v[()]
            else:
                extra[k] = v[()]
        ret.features.append(feats)
        ret.frames.append(frames)
        ret.extras.append(extra)

    if load_attrs:
        import pickle
        try:
            with open(os.path.join(path, 'attrs.pkl'), 'rb') as f:
                attrs = pickle.load(f)
        except IOError:
            attrs = {}
        return ret, attrs
    else:
        return ret
