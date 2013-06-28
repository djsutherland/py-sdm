from __future__ import division, print_function

from collections import Counter
from contextlib import closing
import os
import sys

import numpy as np

from .utils import (izip, iterkeys, iteritems, lazy_range, strict_zip,
                    str_types, is_integer_type)

_default_category = 'none'
_do_nothing_sentinel = object()

DEFAULT_VARFRAC = 0.7

def _group(boundaries, arr):
    return [arr[boundaries[i-1]:boundaries[i]]
            for i in lazy_range(1, len(boundaries))]


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
            if not passed, uses `default_category` for all of them.
            Should not contain the '/' character; sticking to [-\w\d. ]+
            is safest.
        - default_category (optional, default "none"): the default category
            to use for each object if categories is not passed.
        - names (optional): a name for each object. Should be unique per
            category but may have repeats across categories. Same restrictions
            on characters as categories. If not present, defaults to sequential
            integers.
        - any other keyword argument: interpreted as metadata for each
            object. Lists of scalars are converted to numpy arrays; anything
            else is treated as a numpy object array.

    The `data` attribute is a numpy structured array. Each element corresponds
    to a bag. The datatype elements are 'features' (a reference to a bag of
    features), 'category' (a string), 'name' (a string), as well as any extras.
    '''
    def __init__(self, bags, n_pts=None, categories=None, names=None,
                 default_category=_default_category, **extras):
        if bags is _do_nothing_sentinel:
            return  # special path for from_data

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

                if a.ndim == 1:
                    a = a[None, :]
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
        self._boundaries = np.hstack([[0], np.cumsum(n_pts)])

        n_bags = n_pts.size

        # handle categories
        if categories is None:
            categories = np.repeat(default_category, n_bags)
        else:
            categories = np.asarray(categories, dtype=str)
            if len(categories) != n_bags:
                raise ValueError("have {} bags but {} categories".format(
                    n_bags, len(categories)))

        # handle names
        if names is None:
            names = np.array([str(i) for i in lazy_range(n_bags)])
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
        self._extra_names = frozenset(the_extras)

        # do the vstacking, if necessary
        if still_stack:
            self._features = bags = np.vstack(new_bags)

        # make the structured array containing everything
        dtype = self._get_dtype(categories, names, the_extras)
        self.data = data = np.empty(n_bags, dtype=dtype)

        self._refresh_features()
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

    ############################################################################
    ### Copying, pickling, etc

    @classmethod
    def from_data(cls, data, copy=False, deep=False, _memo=None):
        '''
        Constructs a Features instance from its .data attribute.

        Copies the data if copy=True is passed. Note that this will copy the
        features, but not any extras which are object references. Use deep=True
        in that case.
        '''
        new = cls(_do_nothing_sentinel)
        new._update_from_data(data, copy=copy, deep=deep, _memo=_memo)
        return new

    def _update_from_data(self, data, copy=False, deep=False, _memo=None):
        feats = data['features']
        self._n_pts = np.array([f.shape[0] for f in feats])
        self._boundaries = np.hstack([[0], np.cumsum(self._n_pts)])

        reg_names = frozenset(['category', 'features', 'name'])
        self._extra_names = frozenset(data.dtype.names) - reg_names

        # TODO: avoid copying data (as much as is possible) by examining
        #       feats[i].base. If we're copying or subsetting, we should be
        #       able to be smarter than this.
        self._features = np.vstack(feats)

        if copy:
            if deep:
                from copy import deepcopy

            self.data = d = np.empty_like(data)
            for n in d.dtype.names:
                if n != 'features':
                    d[n] = deepcopy(data[n], _memo) if deep else data[n]
        else:
            self.data = data

        self._refresh_features()

    def _refresh_features(self):
        self.data['features'] = _group(self._boundaries, self._features)

    def __copy__(self):
        return Features.from_data(self.data, copy=True, deep=False)

    def copy(self):
        return self.__copy__()

    def __deepcopy__(self, _memo=None):
        return Features.from_data(self.data, copy=True, deep=True, _memo=_memo)

    def __getstate__(self):
        return (self.data,)

    def __setstate__(self, state):
        data, = state
        self._update_from_data(state, copy=False)

    ############################################################################
    ## General magic methods for basic behavior

    def __repr__(self):
        s = '<Features: {:,} bags with {} {}-dimensional points ({:,} total)>'
        min_p = self._n_pts.min()
        max_p = self._n_pts.max()
        if min_p == max_p:
            pts = "{:,}".format(min_p)
        else:
            pts = '{:,} to {:,}'.format(min_p, max_p)
        return s.format(len(self), pts, self.dim, self.total_points)

    def __len__(self):
        return self.data.size

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        if (isinstance(key, str_types) or
                (isinstance(key, tuple) and any(isinstance(x) for x in key))):
            raise TypeError("Features indexing only subsets rows")

        if np.isscalar(key):
            return self.data[key]
        else:
            return Features.from_data(self.data[key], copy=False)

    def __add__(self, oth):
        if isinstance(oth, Features):
            common_extras = dict((k, np.vstack((self[k], oth[k])))
                                 for k in self._extra_names & oth._extra_names)
            return Features(
                np.vstack((self._features, oth._features)),
                n_pts=np.hstack((self._n_pts, oth._n_pts)),
                categories=np.hstack((self.categories, oth.categories)),
                names=np.hstack((self.names, oth.names)),
                **common_extras)

        if isinstance(oth, list):  # TODO: support np object arrays too?
            feats = np.vstack([self._features] + oth)
            n_pts = np.hstack([self._n_pts] + [len(x) for x in oth])

            oth_cats = np.repeat(_default_category, len(oth))
            cats = np.hstack([self.categories, oth_cats])

            names = [str(i) for i in range(len(feats), len(feats) + len(oth))]
            names.insert(0, self.names)
            names = np.hstack(names)

            return Features(feats, n_pts=n_pts, categories=cats, names=names)

        return NotImplemented

    def __radd__(self, oth):
        if isinstance(oth, list):
            feats = np.vstack(oth + [self._features])
            n_pts = np.hstack([len(x) for x in oth] + [self._n_pts])

            oth_cats = np.repeat(_default_category, len(oth))
            cats = np.hstack([oth_cats, self.categories])

            names = [str(i) for i in range(len(feats), len(feats) + len(oth))]
            names.append(self.names)
            names = np.hstack(names)

            return Features(feats, n_pts=n_pts, categories=cats, names=names)

        return NotImplemented

    ############################################################################
    ### Properties to get at the basic data

    @property
    def total_points(self):
        "The total number of points in all bags."
        return self._features.shape[0]

    @property
    def dim(self):
        "The dimensionality of the features."
        return self._features.shape[1]

    @property
    def dtype(self):
        "The data type of the feature vectors."
        return self._features.dtype

    features = property(lambda self: self.data['features'])
    categories = category = property(lambda self: self.data['category'])
    names = name = property(lambda self: self.data['name'])

    # handle extras too, even though we don't know their names in advance...
    # TODO: actually make these in the constructor, so tab-complete/etc works
    def __getattr__(self, name):
        if name in self._extra_names:
            return self.data[name]
        else:
            return getattr(super(Features, self), name)

    ############################################################################
    ### Adding new extras to an existing object

    def add_extra(self, name, values, dtype=None, inplace=False):
        '''
        Adds a single "extra" value to this Features object.

        See add_extras for details.
        '''
        dtypes = None if dtype is None else [dtype]
        return self.add_extras(names=[name], values=[values], dtypes=dtypes,
                               inplace=inplace)

    def add_extras(self, names, values, dtypes=None, inplace=False):
        '''
        Adds new "extra" values to this Features object.

        Note that for implementation reasons, this requires making a copy of
        the .data array containing all the metadata (though not the actual
        features array itself).

        Arguments:
            - names: a list of names for the new extra values
            - values: a list of the actual values for the new extras. Should
                    be broadcastable to be of shape (len(self),).
            - dtypes (optional): a list of the data types for the new extras.
                    If not passed, uses the dtype of np.asarray(val) for each
                    value. If you don't pass dtypes and values contains
                    objects other than numpy arrays, an extra copy will be
                    made during this process.
            - inplace (optional, default False): if True, adds the extra to
                    this object (though metadata is copied as noted above).
                    If False, returns a new object with the extra added. Note
                    that the new object will be like a shallow copy of this
                    one: the features array and any object-type extras will
                    be shared.
        '''
        # Can't use numpy.lib.recfunctions.append_fields:
        # https://github.com/numpy/numpy/issues/2346
        len_set = set([len(names), len(values)])
        if dtypes is not None:
            len_set.add(len(dtypes))
        if len(len_set) != 1:
            raise ValueError("names, values, and dtypes (if passed) should be "
                             "of same length")

        name_set = set(names)
        if len(name_set) != len(names):
            raise ValueError("can't repeat names...")
        elif not name_set.isdisjoint(self.data.dtype.names):
            raise ValueError("can't use names already in use")

        if dtypes is None:
            values = [np.asarray(val) for val in values]
            dtypes = [val.dtype for val in values]

        old_descr = self.data.dtype.descr
        new_descr = strict_zip(names, dtypes)
        new = np.empty(len(self), dtype=old_descr + new_descr)

        for name, dtype in old_descr:
            new[name] = self.data[name]
        for name, value in izip(names, values):
            new[name] = value

        if inplace:
            self.data = new
            self._extra_names = self._extra_names.union(names)
        else:
            return Features.from_data(new)

    ############################################################################
    ### Transforming the features

    def _replace_bags(self, bags, n_pts=None, inplace=False):
        if n_pts is None:
            n_pts = [b.shape[0] for b in bags]
            bags = np.vstack(bags)
        else:
            bags = np.asarray(bags)
        assert bags.ndim == 2

        if inplace:
            self._n_pts = np.asarray(n_pts)
            self._boundaries = np.hstack([0], np.cumsum(self._n_pts))
            self._features = bags
            self._refresh_features()
        else:
            return self.__class__(
                bags, n_pts=n_pts, categories=self.categories, names=self.names,
                **dict((k, self.data[k]) for k in self._extra_names))

    def _apply_transform(self, transformer, fit_first, inplace=False,
                         dtype=None):
        '''
        Transforms the features using an sklearn-style transformer object that
        should be fit to the full, stacked feature matrix. Assumes that the
        transformer supports the "copy" attribute, and that it does not change
        the number or order of points (though it may change their
        dimensionality).

        transformer: the transformer object
        fit_first: whether to fit the transformer to the objects first
        dtype: fit to the features.astype(dtype) if not None

        By default, returns a new Features instance.
        If inplace is passed, modifies this instance; doesn't return anything.
        '''
        transformer.copy = not inplace

        feats = self._features
        if dtype is not None:
            feats = feats.astype(dtype)

        if fit_first:
            transformed = transformer.fit_transform(feats)
        else:
            transformed = transformer.transform(feats)
        return self._replace_bags(
            transformed, n_pts=self._n_pts, inplace=inplace)


    def pca(self, pca=None, unfit_pca=None,
            k=None, varfrac=DEFAULT_VARFRAC, randomize=False, whiten=False,
            dtype=None,
            ret_pca=False, inplace=False):
        '''
        Runs the features through principal components analysis to reduce their
        dimensionality.

        By default, returns a new Features instance.
        If inplace is passed, modifies this instance; doesn't return anything.
        If ret_pca is passed: returns the PCA object as well as whatever else
                              it would have returned.

        If `pca` is passed, uses that pre-fit PCA object to transform. This is
        useful for transforming test objects consistently with training objects.

        Otherwise, if `unfit_pca` is passed, that object's fit_transform()
        method is called to fit the samples and transform them.

        Otherwise, the following options specify which type of PCA to perform:
            k: a dimensionality to reduce to. Default: use varfrac instead.

            varfrac: the fraction of variance to preserve. Overridden by k.
                Default: 0.7. Can't be used for randomized or sparse PCA.

            randomize: use a randomized PCA implementation. Default: no.

            whiten: whether to whiten the inputs, removing linear correlations
                    across features

            dtype: the dtype of the feature matrix to use.
        '''
        # figure out what PCA instance we should use
        if pca is not None:
            fit_first = False
        elif unfit_pca is not None:
            pca = unfit_pca
            fit_first = True
        else:
            from sklearn.decomposition import PCA, RandomizedPCA
            fit_first = True

            if k is None:
                if randomize:
                    raise ValueError("can't randomize without a specific k")
                pca = PCA(varfrac, whiten=whiten)
            else:
                pca = (RandomizedPCA if randomize else PCA)(k, whiten=whiten)

        r = self._apply_transform(pca, fit_first=fit_first, inplace=inplace)
        if ret_pca:
            return pca if inplace else (r, pca)
        else:
            return r

    def standardize(self, scaler=None, ret_scaler=False, inplace=False,
                    cast_dtype=np.float32):
        '''
        Standardizes the features so that each dimension has zero mean and unit
        variance.

        By default, returns a new Features instance.
        If inplace is passed, modifies this instance; doesn't return anything.
        If ret_scaler is passed: returns the scaler object as well as whatever
                                 else it would have returned.
        If cast_dtype is not None, casts non-float data arrays to this dtype
        first.

        If `scaler` is passed, uses that pre-fit scaler to transform. This is
        useful for transforming test objects consistently with training objects.
        '''
        fit_first = False
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            fit_first = True

        kw = {'fit_first': fit_first, 'inplace': inplace}
        if self._features.dtype.kind != 'f':
            kw['dtype'] = cast_dtype
        r = self._apply_transform(scaler, **kw)
        if ret_scaler:
            return scaler if inplace else (r, scaler)
        else:
            return r

    def normalize(self, norm='l2', inplace=False, cast_dtype=np.float32):
        '''
        Normalizes the features so that each vector has unit norm (l1 or l2).

        By default, returns a new Features instance.
        If inplace is passed, modifies this instance; doesn't return anything.
        If cast_dtype is not None, casts non-float data arrays to this dtype
        first.
        norm: 'l2' (default) or 'l1'.

        This transformation is stateless, so unlike pca() or standardize()
        there's no point in returning the normalizer object.
        '''
        from sklearn.preprocessing import Normalizer
        normalizer = Normalizer(norm)

        dtype = None if self.dtype.kind == 'f' else cast_dtype

        return self._apply_transform(
            normalizer, fit_first=False, inplace=inplace, dtype=dtype)

    def condense_kmeans(self, n_clusters, max_iter=20, inplace=False,
                        progressbar=False, cast_dtype=np.float32):
        '''
        Condenses the number of points in a sample set through k-means.
        '''
        from sklearn.cluster import MiniBatchKMeans

        dtype = None if self.dtype.kind == 'f' else cast_dtype

        # TODO: use full batch kmeans for small sizes?
        # TODO: use elkan k-means from vlfeat instead?

        # most of the work is parallelized by MKL. still, not super fast.
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, init='random', compute_labels=False,
            max_iter=max_iter)

        new_bags = []
        feats_iter = self.features
        if progressbar:
            from mp_utils import progress
            feats_iter = progress()(feats_iter)
        for bag in feats_iter:
            if bag.shape[0] <= n_clusters:
                new_bags.append(bag)
            else:
                if dtype is not None:
                    bag = bag.astype(dtype)
                kmeans.fit(bag)
                new_bags.append(kmeans.cluster_centers_)

        return self._replace_bags(new_bags, inplace=inplace)

    ############################################################################
    ### generic I/O helpers

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

    def save(self, path, format='hdf5', **attrs):
        '''
        Saves into an output file. Calls save_as_hdf5 if format=='hdf5'
        (the default), or save_as_perbag if format=='perbag'.
        '''
        if format == 'hdf5':
            return self.save_as_hdf5(path, **attrs)
        elif format == 'perbag':
            return self.save_as_perbag(path, **attrs)
        else:
            raise TypeError("unknown save format '{}'".format(format))

    @classmethod
    def load(cls, path, **kwargs):
        '''
        Loads from either an hdf5 file or a perbag directory. Calls
        load_from_hdf5 if path is a file or load_from_perbag if a directory.
        '''
        if os.path.isdir(path):
            return cls.load_from_perbag(path, **kwargs)
        else:
            return cls.load_from_hdf5(path, **kwargs)

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
        with h5py.File(filename, 'w') as f:
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
            boundaries = np.hstack([[0], np.cumsum(n_pts)])

            # allocate space for features and extras
            # TODO: go straight into a data array, save a copy...
            if features_dtype is None:
                dtype = dtypes.pop()
                if dtypes:
                    raise TypeError("different features have different dtypes")
                    # TODO: find a dtype that'll cover all of them
            else:
                dtype = features_dtype
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
                features[boundaries[i]:boundaries[i+1]] = g['features']

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
        if not os.path.exists(path):
            os.makedirs(path)

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
        another list of that format. functools.partial(random.sample, k=100) can
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
            npz_path = os.path.join(path, cat, fname + '.npz')
            with closing(np.load(npz_path)) as data:
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
