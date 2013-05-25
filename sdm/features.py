from __future__ import division, print_function

from collections import Counter
import os
import sys

import numpy as np

from .utils import izip, iterkeys, iteritems, is_integer_type

_default_category = 'none'

class Features(object):
    '''
    A wrapper class for storing bags of features. (A *bag* is a set of feature
    vectors corresponding to a single object.)

    Stores them stacked into a single array (to make e.g. PCA and nearest-
    neighbor searches easier), but allows seamless access to individual sets.

    Also stores some metadata corresponding to each bag (e.g. labels, names).

    To create a Features object, pass:
        - The features. You can do this in one of two ways:
            - pass bags as a list of numpy arrays, one per object, whose
                dimensionality should be n_pts x dim. n_pts can vary for each
                bag, but dim must be the same for all bags. n_pts cannot be 0.
            - pass bags as a single numpy array of shape sum(n_pts) x dim, and
                also pass n_pts as an array-like object containing the number
                of points in each bag. This should be a list of positive
                integers whose sum is equal to the number of rows in bags.
                bags will be C-ordered.
        - categories (optional): a list of the "category" for each object. If
            passed, should be of equal length to the number of bags. This
            might be a class name, a data source, etc. Used in storing the data;
            if not passed, uses "none" for all of them. Should not contain the
            '/' character.
        - names (optional): a name for each object. Should be unique per
            category but may have repeats across categories. Should not contain
            the '/' character. If not present, defaults to sequential integers.
        - any other keyword argument: interpreted as a label metadata for each
            object. Should be a list of scalars or strings, one per object.

    The `data` attribute is a numpy structured array containing
    '''
    def __init__(self, bags, n_pts=None, categories=None, names=None, **extras):
        # load the features
        if n_pts is not None:
            n_pts = np.squeeze(n_pts)
            if n_pts.ndim != 1:
                raise TypeError("n_pts must be 1-dimensional")
            if n_pts.size == 0:
                raise TypeError("must have at least one bag")
            if np.any(n_pts <= 0):
                raise TypeError("n_pts must all be positive")

            if not is_integer_type(n_pts):
                rounded = np.rint(n_pts)
                if all(rounded == n_pts):
                    n_pts = rounded.astype(int)
                else:
                    raise TypeError("n_pts must be integers")

            bags = np.asarray(bags, order='C')
            if bags.ndim != 2 or bags.shape[0] != np.sum(n_pts):
                raise TypeError("bags must have shape sum(n_pts) x dim")
            if bags.shape[1] == 0:
                raise TypeError("bags must have dimension > 0")

            self._features = bags
            still_stack = False
        else:
            if len(bags) == 0:
                raise ValueError("must have at least one bag")

            dim = None
            new_bags = []
            n_pts = []
            for bag in bags:
                a = np.asarray(bag, order='C')

                if a.ndim != 2:
                    raise TypeError("each bag must be n_pts x dim")

                if dim is None:
                    dim = a.shape[1]
                elif a.shape[1] != dim:
                    raise TypeError("bags' second dimension must be consistent")

                if a.shape[0] == 0:
                    raise TypeError("each bag must have at least one point")

                if a.dtype.kind not in 'fiu':
                    raise TypeError("can't handle type {}".format(a.dtype.name))

                new_bags.append(a)
                n_pts.append(a.shape[0])

            n_pts = np.asarray(n_pts)
            still_stack = True
            # delay doing the actual vstack until later, because that can take
            # a while and is wasted if there's an error in one of the other
            # arguments

        self._n_pts = n_pts
        self._end_pts = np.cumsum(n_pts)
        self._start_pts = np.hstack(([0], self._end_pts[:-1]))

        n_bags = n_pts.size

        # handle categories
        if categories is None:
            categories = np.repeat(_default_category, n_bags)
        else:
            categories = np.asarray(categories, dtype=str)
            if len(categories) != n_bags:
                raise ValueError("have {} bags but {} categories".format(
                    n_bags, len(categories)))

        # handle names
        if names is None:
            names = np.arange(n_bags).astype(str)
        else:
            names = np.asarray(names, dtype=str)
            if len(names) != n_bags:
                raise ValueError("have {} bags but {} names".format(
                    n_bags, len(names)))

            # check that they're unique per category
            cat_names = np.zeros(n_bags,
                dtype=[('cat', categories.dtype), ('name', names.dtype)])
            cat_names['cat'] = categories
            cat_names['name'] = names
            if np.unique(cat_names).size != n_bags:
                raise ValueError("category/name pairs must be unique")

        # handle extras
        the_extras = {}
        for name, vals in iteritems(extras):
            if len(vals) != n_bags:
                raise ValueError("have {} bags but {} values for {}".format(
                    n_bags, len(vals), name))
            the_extras[name] = np.asarray(vals)
        self._extra_names = set(the_extras)

        # do the vstacking, if necessary
        if still_stack:
            self._features = bags = np.vstack(new_bags)

        # make the structured array containing everything
        self.dtype = dtype = self._get_dtype(categories, names, the_extras)
        self.data = data = np.empty(n_bags, dtype=dtype)

        data['features'] = [bags[start:end] for start, end
                            in izip(self._start_pts, self._end_pts)]
        data['category'] = categories
        data['name'] = names
        for name, vals in iteritems(the_extras):
            data[name] = vals

    def _get_dtype(self, categories, names, extras):
        dt = [
            ('features', object),
            ('category', categories.dtype),
            ('name', names.dtype)
        ]
        # in python 2 only, have to encode the names...sigh.
        if sys.version_info.major == 2:
            dt += [(n.encode(), vals.dtype) for n, vals in iteritems(extras)]
        else:
            dt += [(n, vals.dtype) for n, vals in iteritems(extras)]
        return dt

    def __repr__(self):
        s = '<Features: {} bags each with {} {}-dimensional points ({} total)>'
        min_p = self._n_pts.min()
        max_p = self._n_pts.max()
        pts = min_p if min_p == max_p else '{} to {}'.format(min_p, max_p)
        return s.format(len(self), pts, self.dim, self.total_points)

    @property
    def total_points(self):
        return self._features.shape[0]

    @property
    def dim(self):
        return self._features.shape[1]

    ### indexing/etc works directly on the data array
    def __getitem__(self, key): return self.data[key]
    def __len__(self): return self.data.size
    def __iter__(self): return iter(self.data)

    ### convenience properties to get at a single column of the data
    features = property(lambda self: self.data['features'])
    categories = category = property(lambda self: self.data['category'])
    names = name = property(lambda self: self.data['name'])

    # handle extras too, even though we don't know their names in advance...
    def __getattr__(self, name):
        if name in self._extra_names:
            return self.data[name]
        else:
            return super(Features, self).__getattr__(name)

    ### I/O helper
    @staticmethod
    def _missing_extras(dtype):
        """
        For arrays with dtype with missing vals: returns new dtype, default val.
        """
        if dtype.kind in 'fc':  # float/complex types
            return dtype, np.nan
        elif dtype.kind in 'O':  # object types
            return dtype, None
        elif dtype.kind in 'aSU':  # string types
            return dtype, ''
        elif dtype.kind in 'biu':  # integer types: no missing type, so switch
                                   # to float and use nan
            return np.float, np.nan
        else:  # other types: no default, so switch to object type and use None
            return object, None

    ############################################################################
    ### Stuff relating to hdf5 feature files

    def save_as_hdf5(self, filename, **attrs):
        '''
        Saves into an HDF5 file.

        Also saves any keyword args as a dateset under '/meta'.

        Each bag is saved as "features" and "frames" in /category/filename; any
        "extras" get added there as a (probably scalar) dataset named by the
        extra's name.
        '''
        import h5py
        with h5py.File(filename) as f:
            skip_set = frozenset(['category', 'name'])
            for row in self:
                g = f.require_group(row['category']).create_group(row['name'])
                for name, val in izip(row.dtype.names, row):
                    if name not in skip_set:
                        g[name] = val

            meta = f.require_group('_meta')
            for k, v in iteritems(attrs):
                meta[k] = v

    @classmethod
    def load_from_hdf5(cls, filename, load_attrs=False, features_dtype=None,
                       cats=None, pairs=None, subsample_fn=None,
                       names_only=False):
        '''
        Reads a Features instance from an h5py file created by save_features().

        If load_attrs, also returns a dictionary of meta values loaded from
        root attributes, '/_meta' attributes, '/_meta' datasets.

        features_dtype specifies the datatype to load features as.

        If cats is passed, only load those with a category in cats (as checked
        by the `in` operator, aka the __contains__ special method).

        If pairs is passed, tuples of (category, name) are checked with the `in`
        operator. If cats is also passed, that check applies first.

        subsample_fn is applied to a list of (category, name) pairs, and returns
        another list of that format. functools.partial(random.sample, k=100) can
        be used to subsample 100 bags unifornmly at random, for example.

        If names_only is passed, the list of (category, name) pairs is returned
        without having loaded any data. load_attrs is also ignored.
        '''
        import h5py

        with h5py.File(filename, 'r') as f:
            bag_names = []
            for cat, cat_g in iteritems(f):
                if cat != '_meta' and (cats is None or cat in cats):
                    for fname in iterkeys(cat_g):
                        if pairs is None or (cat, fname) in pairs:
                            bag_names.append((cat, fname))

            if subsample_fn is not None:
                bag_names = subsample_fn(bag_names)

            if names_only:
                return bag_names

            # first pass: get numbers/type of features, names/types of metadata
            dim = None
            n_pts = []
            dtypes = set()
            extra_types = {}
            with_extras = Counter()

            for cat, fname in bag_names:
                g = f[cat][fname]

                feats = g['features']
                shape = feats.shape
                if len(shape) != 2:
                    msg = "malformed file: {}/{}/features has shape {}"
                    raise ValueError(msg.format(cat, fname, shape))
                elif shape[0] == 0:
                    msg = "malformed file: {}/{} has no features"
                    raise ValueError(msg.format(cat, fname))

                if dim is None:
                    dim = shape[1]
                elif shape[1] != dim:
                    msg = "malformed file: {}/{} has feature dim {}, expected {}"
                    raise ValueError(msg.format(cat, fname, shape[1], dim))

                n_pts.append(feats.shape[0])
                dtypes.add(feats.dtype)

                for name, val in iteritems(g):
                    if name == 'features':
                        continue

                    dt = val.dtype if all(s == 1 for s in val.shape) else object
                    if name not in extra_types:
                        extra_types[name] = dt
                    elif extra_types[name] != dt:
                        msg = "different {}s have different dtypes"
                        raise TypeError(msg.format(name))
                        # TODO: find a dtype that'll cover all of them
                    with_extras[name] += 1

            n_bags = len(bag_names)
            n_pts = np.asarray(n_pts)
            end_pts = np.cumsum(n_pts)
            start_pts = np.hstack([0, end_pts[:-1]])

            # allocate space for features and extras
            dtype = dtypes.pop()
            if dtypes:
                raise TypeError("different features have different dtypes")
                # TODO: find a dtype that'll cover all of them
            features = np.empty((n_pts.sum(), dim), dtype=dtype)

            extras = {}
            extra_defaults = {}
            for name, dt in iteritems(extra_types):
                if with_extras[name] != n_bags:
                    dt, d = cls._missing_extras(dt)
                    extra_defaults[name] = d
                    print("WARNING: {} missing values for {}. using {} instead"
                            .format(n_bags - with_extras[name], name, d),
                          file=sys.stderr)
                extras[name] = np.empty(n_bags, dtype=dt)

            # actually load all the features and extras
            for i, (cat, fname) in enumerate(bag_names):
                g = f[cat][fname]
                features[start_pts[i]:end_pts[i]] = g['features']

                for ex_name in extra_types:
                    if ex_name in g:
                        extras[ex_name][i] = g[ex_name][()]
                    else:
                        extras[ex_name][i] = extra_defaults[ex_name]

            categories, names = zip(*bag_names)
            obj = cls(features, n_pts=n_pts, categories=categories, names=names,
                      **extras)

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
                return obj, attrs

            return obj

    ############################################################################
    ### Stuff relating to per-bag npz feature files

    def save_as_perbag(self, path, **attrs):
        '''
        Save into one npz file for each bag, named like
            path/category/name.npz

        Also saves any extra attributes passed as keyword arguments in
            path/attrs.pkl
        '''
        import pickle
        with open(os.path.join(path, 'attrs.pkl'), 'wb') as f:
            pickle.dump(attrs, f)

        skip_set = frozenset(['category', 'name'])
        for row in self:
            dirpath = os.path.join(path, row['category'])
            if not os.path.isdir(dirpath):
                os.mkdir(dirpath)

            data = dict((k, v) for k, v in izip(row.dtype.names, row)
                        if k not in skip_set)
            np.savez(os.path.join(dirpath, row['name'] + '.npz'), **data)

    @classmethod
    def load_from_perbag(cls, path, load_attrs=False, features_dtype=None,
                         cats=None, pairs=None, subsample_fn=None,
                         names_only=False):
        '''
        Reads a Features instance from a directory of npz files created
        by save_as_perbag().

        If load_attrs, also returns a dictionary of meta values loaded from the
        `attrs.pkl` file, if it exists.

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

        # Unlike in read_from_hdf5, we can't just get the size of arrays without
        # loading them in. So we do the vstacking thing.

        bags = []
        extras = []  # starts as a list of dictionaries. will change to a dict
                     # of arrays after we load them all in.
        extra_types = {}
        with_extras = Counter()

        for cat, fname in bag_names:
            data = np.load(os.path.join(path, cat, fname + '.npz'))
            feats = None
            extra = {}
            for k, v in iteritems(data):
                if k == 'features':
                    if features_dtype is not None:
                        feats = np.asarray(v, dtype=features_dtype)
                    else:
                        feats = v[()]
                else:
                    dt = v.dtype if all(s == 1 for s in v.shape) else object
                    if k not in extra_types:
                        extra_types[k] = dt
                    elif extra_types[k] != dt:
                        msg = "different {}s have different dtypes"
                        raise TypeError(msg.format(k))
                        # TODO: find a dtype that'll cover all of them
                    extra[k] = v[()]
                    with_extras[k] += 1
            bags.append(feats)
            extras.append(extra)

        # post-process the extras
        n_bags = len(bags)
        the_extras = {}
        extra_defaults = {}

        for name, dt in iteritems(extra_types):
            if with_extras[name] != n_bags:
                dt, d = cls._missing_extras(dt)
                extra_defaults[name] = d
                print("WARNING: {} missing values for {}. using {} instead"
                        .format(n_bags - with_extras[name], name, d),
                      file=sys.stderr)
            the_extras[name] = np.empty(n_bags, dtype=dt)

        for i, extra_d in enumerate(extras):
            for name, default in iteritems(extra_defaults):
                the_extras[name][i] = extra_d.get(name, default)

        categories, names = zip(*bag_names)
        obj = cls(bags, categories=categories, names=names, **the_extras)

        if load_attrs:
            import pickle
            try:
                with open(os.path.join(path, 'attrs.pkl'), 'rb') as f:
                    attrs = pickle.load(f)
            except IOError:
                attrs = {}
            return obj, attrs
        else:
            return obj
