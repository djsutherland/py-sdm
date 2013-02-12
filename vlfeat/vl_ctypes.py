from ctypes import (cdll, sizeof,
    c_float, c_double,
    c_int, c_int8, c_int32, c_int64,
    c_uint, c_uint8, c_uint32, c_uint64)
from ctypes.util import find_library

import numpy as np

_loc = find_library('vl')
LIB = cdll[_loc if _loc is not None else 'libvl.so']
# TODO - more options for finding the library

# TODO actually figure out if it's built LP64 or ILP64 or whatever
vl_size = c_uint64
vl_index = c_int64

np_to_c_types = {}
c_to_np_types = {}
_dtypes = {
    'i': (c_int, c_int8, c_int32, c_int64),
    'u': (c_uint, c_uint8, c_uint32, c_uint64),
    'f': (c_float, c_double),
}
for t, c_types in _dtypes.items():
    for c_type in c_types:
        dtype = np.dtype('<{}{}'.format(t, sizeof(c_type)))
        np_to_c_types[dtype] = c_type
        c_to_np_types[c_type] = dtype
del t, c_types, c_type, dtype

vl_epsilon_f = 2 ** -23  # smallest representable single
