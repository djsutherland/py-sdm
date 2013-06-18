from __future__ import division, print_function

import functools
import itertools
from operator import methodcaller
import os
import shutil
import sys

import numpy as np

### stuff for python 2/3 compatability; could just use six, but...whatever

if sys.version_info.major == 2:
    izip = itertools.izip
    strict_zip = zip
    imap = itertools.imap
    strict_map = map
    lazy_range = xrange
    str_types = (basestring, str, unicode)
    bytes = str
    raw_input = raw_input
    iterkeys = methodcaller('iterkeys')
    itervalues = methodcaller('itervalues')
    iteritems = methodcaller('iteritems')
else:
    izip = zip
    imap = map
    lazy_range = range
    str_types = (str,)
    bytes = bytes
    raw_input = input
    iterkeys = methodcaller('keys')
    itervalues = methodcaller('values')
    iteritems = methodcaller('items')

    @functools.wraps(map)
    def strict_zip(*args, **kwargs):
        return list(zip(*args, **kwargs))

    @functools.wraps(map)
    def strict_map(*args, **kwargs):
        return list(map(*args, **kwargs))


def identity(x):
    return x


### some little numpy utilities
eps = np.spacing(1)

def col(a): return a.reshape((-1, 1))
def row(a): return a.reshape((1, -1))
def get_col(X, c): return X[:, c].ravel()


def is_integer_type(x):
    return issubclass(np.asanyarray(x).dtype.type, np.integer)
def is_categorical_type(ary):
    ary = np.asanyarray(ary)
    return is_integer_type(ary) or ary.dtype.kind == 'b'

def is_integer(x):
    return np.isscalar(x) and is_integer_type(x)
def is_categorical(x):
    return np.isscalar(x) and is_categorical_type(x)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


### type-checkers for argparse arguments that verify constraints

def positive_int(val):
    val = int(val)
    if val <= 0:
        raise TypeError("must be a positive integer")
    return val

def nonnegative_int(val):
    val = int(val)
    if val < 0:
        raise TypeError("must be a nonnegative integer")
    return val

def positive_float(val):
    val = float(val)
    if val <= 0:
        raise TypeError("must be a positive number")
    return val

def nonnegative_float(val):
    val = float(val)
    if val < 0:
        raise TypeError("must be a nonnegative number")
    return val

def portion(val):
    val = float(val)
    if not 0 <= val <= 1:
        raise TypeError("must be a number between 0 and 1")
    return val


def confirm_outfile(filename, dir=False):
    '''
    Check that a file doesn't exist, prompt if it does, and check it's writable.
    Calls sys.exit() if not.
    '''
    if os.path.exists(filename):
        if os.path.isdir(filename):
            msg = "Output {} '{}' is already a directory! We'll delete it."
        else:
            msg = "Output {} '{}' already exists; will be deleted."
        msg += " Continue? [yN] "
        resp = raw_input(msg.format('dir' if dir else 'file', filename))

        if not resp.lower().startswith('y'):
            sys.exit("Aborting.")

        if os.path.isdir(filename):
            shutil.rmtree(filename)

    if dir:
        os.makedirs(filename)
        filename = os.path.join(filename, 'test')

    try:
        with open(filename, 'w'):
            pass
        os.remove(filename)
    except Exception as e:
        sys.exit("{}: can't write to '{}'".format(e, filename))


def get_status_fn(val):
    if val is True:
        return functools.partial(print, file=sys.stderr)
    elif val in (None, False):
        return lambda *args, **kwargs: None
    else:
        return val


def read_cell_array(f, data, dtype=None):
    return [
        np.ascontiguousarray(np.transpose(f[ptr]), dtype=dtype)
        for row in data
        for ptr in row
    ]
