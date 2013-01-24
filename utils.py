from __future__ import division, print_function

import functools
import itertools
import os
import sys

import numpy as np

### stuff for python 2/3 compatability; could just use six, but...whatever

if sys.version_info.major == 2:
    izip = itertools.izip
    imap = itertools.imap
    strict_map = map
    lazy_range = xrange
    str_types = (basestring, str, unicode)
    raw_input = raw_input
else:
    izip = zip
    imap = map
    lazy_range = range
    str_types = (str,)
    raw_input = input

    @functools.wraps(map)
    def strict_map(*args, **kwargs):
        return list(map(*args, **kwargs))


### some little numpy utilities
eps = np.spacing(1)

def col(a): return a.reshape((-1, 1))
def row(a): return a.reshape((1, -1))
def get_col(X, c): return X[:, c].ravel()

def is_integer_type(x):
    return issubclass(np.asanyarray(x).dtype.type, np.integer)

def is_integer(x):
    return np.isscalar(x) and is_integer_type(x)


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


def confirm_outfile(filename):
    '''
    Check that a file doesn't exist, prompt if it does, and check it's writable.
    Calls sys.exit() if not.
    '''
    if os.path.exists(filename):
        resp = raw_input("Output file '{}' already exists; will be deleted. "
                         "Continue? [yN] ".format(filename))
        if not resp.lower().startswith('y'):
            sys.exit("Aborting.")
    try:
        with open(filename, 'w'):
            pass
        os.remove(filename)
    except Exception as e:
        sys.exit("{}: can't write to '{}'".format(e, filename))
