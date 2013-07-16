from cStringIO import StringIO
from contextlib import closing
from functools import partial

import numpy as np

import cyflann
from ..typedbytes_utils import tb, register_read, register_write

# TODO: test with non-seekable files


def _check_all_eq(expected, real):
    assert np.all(expected == real)

def test_typedbytes_ndarray():
    for ary in [np.arange(30, dtype=np.uint8),
                np.random.normal(size=(10, 12, 100)),
                np.linspace(1, 3.2, 121).astype(np.float128)]:
        with closing(StringIO()) as sio:
            out = tb.Output(sio)
            register_write(out)
            out.write(ary)

            sio.seek(0)
            inp = tb.Input(sio)
            register_read(inp)
            ary2 = inp.read()

            yield partial(_check_all_eq, ary, ary2)


def _check_flann(idx1, idx2):
    assert np.all(idx1.data == idx2.data)
    assert np.allclose(idx1.nn_index(idx1.data)[1], idx2.nn_index(idx1.data)[1])

def test_typedbytes_flann():
    pts = np.random.normal(size=(100, 2))
    for algorithm in ['kdtree_single', 'linear']:
        idx = cyflann.FLANNIndex(algorithm=algorithm)
        idx.build_index(pts)

        with closing(StringIO()) as sio:
            out = tb.Output(sio)
            register_write(out)
            out.write(idx)

            sio.seek(0)
            inp = tb.Input(sio)
            register_read(inp)
            idx2 = inp.read()

            fn = partial(_check_flann, idx, idx2)
            fn.description = "flann typedbytes io - {}".format(algorithm)
            yield fn
