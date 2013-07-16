from cStringIO import StringIO
from contextlib import closing
import errno
import os
from shutil import copyfileobj
import struct
import tempfile

import numpy as np
import numpy.lib.format as npy

import cyflann

try:
    import ctypedbytes as tb
except:
    import typedbytes as tb


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


################################################################################
### Typedbytes handling for numpy arrays.
### This is faster than pickle if you're doing it directly to a file, and more
### importantly avoids making an unnecessary copy.

NUMPY_CODE = 0xaa


def _npy_size(ary):
    assert not ary.dtype.hasobject
    magic_len = npy.MAGIC_LEN

    # TODO: could calculate this directly
    with closing(StringIO()) as sio:
        npy.write_array_header_1_0(sio, npy.header_data_from_array_1_0(ary))
        header_len = sio.tell()

    data_len = ary.dtype.itemsize * ary.size

    return magic_len + header_len + data_len


def _file_is_seekable(f):
    try:
        f.tell()
    except IOError as e:
        if e.errno == errno.ESPIPE and e.strerror == 'Illegal seek':
            return False
        raise
    else:
        return True

def check_seekable(the_obj):
    if not hasattr(the_obj, '_file_seekable'):
        the_obj._file_seekable = _file_is_seekable(the_obj.file)


def read_ndarray(f, file_is_seekable=None):
    length, = struct.unpack('>i', f.read(4))
    if length < 0:
        raise ValueError(r"bad length value {}".format(length))

    if file_is_seekable is None:
        file_is_seekable = _file_is_seekable(f)

    if file_is_seekable:
        return np.load(f)
    else:
        with closing(StringIO()) as sio:
            sio.write(f.read(length))
            sio.seek(0)
            return np.load(sio)

def _read_ndarray_in(self):
    seekable = getattr(self, '_file_seekable', None)
    return read_ndarray(self.file, file_is_seekable=seekable)

def register_read_ndarray(input_object):
    check_seekable(input_object)
    input_object.register(NUMPY_CODE, _read_ndarray_in)


def write_ndarray(f, ary):
    f.write(struct.pack('>B', NUMPY_CODE))
    f.write(struct.pack('>i', _npy_size(ary)))
    np.save(f, ary)

def _write_ndarray_out(self, ary):
    write_ndarray(self.file, ary)

def register_write_ndarray(output_object):
    output_object.register(np.ndarray, _write_ndarray_out)


################################################################################
### Typedbytes handling of numpy scalar types

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
### Typedbytes handling of cyflann.FLANNIndex
#
# flann indices are serialized in typedbytes as:
#   - the one-byte type code FLANN_CODE
#   - an int giving the total size
#   - a numpy array of points (npy type code, size, then the .npy bytes)
#   - a bytestring (size, then the save_index() bytes)
FLANN_CODE = 0xab

DEFAULT_TEMPDIR = None
_not_passed = object()

def flann_from_typedbytes(inp, tempdir=_not_passed):
    if tempdir is _not_passed:
        tempdir = DEFAULT_TEMPDIR

    register_read_ndarray(inp)
    length = inp.read_int()
    pts = inp.read()
    index_bytes = inp.read_bytestring()

    with tempfile.NamedTemporaryFile(dir=tempdir) as f:
        f.write(index_bytes)
        f.flush()
        del index_bytes
        index = cyflann.FLANNIndex()
        index.load_index(f.name, pts)

    return index

def register_read_flann(input_object):
    register_read_ndarray(input_object)
    input_object.register(FLANN_CODE, flann_from_typedbytes)


def flann_to_typedbytes(out, index, tempdir=_not_passed):
    if tempdir is _not_passed:
        tempdir = DEFAULT_TEMPDIR

    npy_size = _npy_size(index.data)

    with tempfile.NamedTemporaryFile(dir=tempdir) as f:
        index.save_index(f.name)
        f.seek(0, os.SEEK_END)
        index_size = f.tell()
        f.seek(0)

        out.file.write(struct.pack('>B', FLANN_CODE))
        out.file.write(struct.pack('>i',
            (1 + 4 + npy_size) + (4 + index_size)))
        # npy code, npy size int, npy, index size, index

        out.write(index.data)

        out.file.write(struct.pack('>i', index_size))
        copyfileobj(f, out.file)

def register_write_flann(output_object):
    register_write_ndarray(output_object)
    output_object.register(cyflann.FLANNIndex, flann_to_typedbytes)


################################################################################

def register_read(inp):
    register_read_flann(inp)  # does read_ndarray itself

def register_write(out):
    register_write_flann(out)  # does write_ndarray itself
    register_np_writes(out)
