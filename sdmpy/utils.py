'''
Various helpful utility functions.
'''
from __future__ import division

import numpy as np

eps = np.spacing(1)

def col(a): return a.reshape((-1, 1))
def row(a): return a.reshape((1, -1))
def get_col(X, c): return X[:,c].ravel()

def is_integer(x):
    return np.isscalar(x) and \
            issubclass(np.asanyarray(x).dtype.type, np.integer)
