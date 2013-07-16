from cStringIO import StringIO
from contextlib import closing
import errno
from functools import partial
import struct

import numpy as np
import numpy.lib.format as npy

try:
    import ctypedbytes as tb
except:
    import typedbytes as tb


################################################################################
### Typedbytes handling for numpy arrays.
### This is faster than pickle if you're doing it directly to a file, and more
### importantly avoids making an unnecessary copy.

NUMPY_CODE = 0xaa


def _npy_size(ary):
    magic_len = npy.MAGIC_LEN

    # TODO: could calculate this directly
    with closing(StringIO()) as sio:
        npy.write_array_header_1_0(sio, npy.header_data_from_array_1_0(ary))
        header_len = sio.tell()

    data_len = ary.dtype.itemsize * ary.size

    return magic_len + header_len + data_len


def read_ndarray(f):
    length, = struct.unpack('>i', f.read(4))
    if length < 0:
        raise ValueError(r"bad length value {}".format(length))

    if f.seekable():
        return np.load(f)
    else:
        with closing(StringIO()) as sio:
            sio.write(f.read(length))
            sio.seek(0)
            return np.load(sio)

def _read_ndarray_in(self):
    return read_ndarray(self.file)

def register_read_ndarray(input_object):
    input_object.register(NUMPY_CODE, _read_ndarray_in)


def write_ndarray(f, ary):
    f.write(struct.pack('>B', NUMPY_CODE))
    f.write(struct.pack('>i', _npy_size(ary)))
    np.save(f, ary)

def _write_ndarray_out(self, ary):
    write_ndarray(self.file, ary)

def register_write_ndarray(output_object):
    output_object.register(np.ndarray, _write_ndarray_out)


def register_np_writes(output_object):
    def r(typ, f_name):
        output_object.register(typ, getattr(output_object.__class__, f_name))
    # TODO: should int16, etc get saved as an int, pickled, or (probably best)
    #       made into a custom-width format and saved like that?
    r(np.bool_, 'write_bool')
    r(np.bool8, 'write_bool')
    r(np.int32, 'write_int')
    r(np.int64, 'write_long')
    r(np.float32, 'write_float')
    r(np.float64, 'write_double')
    r(np.string_, 'write_string')


################################################################################


# reads the format you get from http://stackoverflow.com/a/15172498/344821
class SequenceFileInput(tb.Input):
    def read(self):
        # TODO: check the lengths?
        try:
            key_length = self.read_int()
        except struct.error:
            return None

        try:
            key = self._read()
            assert self.file.read(1) == b'\t'
            val_length = self.read_int()
            val = self._read()
            assert self.file.read(1) == b'\n'
        except (StopIteration, struct.error):
            raise struct.error("EOF before complete pair read")

        return key, val
