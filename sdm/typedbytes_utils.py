from cStringIO import StringIO
from contextlib import closing
import errno
import os
from shutil import copyfileobj
import struct
import tempfile
import warnings

import numpy as np
import numpy.lib.format as npy

import cyflann

try:
    import ctypedbytes as tb
except:
    import typedbytes as tb


class TypedbytesSequenceFileStreamingInput(tb.Input):
    '''
    Reads the format that's passed to hadoop streaming jobs when the input is
    a SequenceFile obtained via "loadtb" from a typedbytes file, and you used
    -inputformat org.apache.hadoop.mapred.SequenceFileAsBinaryInputFormat
    (see http://stackoverflow.com/a/15172498/344821).
    '''
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

    def reads(self):
        return iter(self.read, None)
    __iter__ = reads

try:
    from hadoop.io import SequenceFile
except ImportError as e:
    msg = """Couldn't import python-hadoop: {}
If you want to read raw sequence files, install it from
https://github.com/matteobertozzi/Hadoop/tree/master/python-hadoop"""
    warnings.warn(msg)
else:
    import typedbytes  # ctypedbytes segfaults on these pseudo-file-likes

    class TypedbytesSequenceFileInput(object):
        '''
        Reads a raw hadoop SequenceFile containing typedbytes keys and values.
        '''

        def __init__(self, path, start=0, length=0):
            self.reader = SequenceFile.Reader(path)

        def _get_reader(self, f):
            inp = typedbytes.Input(f)
            register_read(inp)
            return inp

        def read(self):
            raw_key = self.reader.nextRawKey()
            if raw_key is None:
                return None
            raw_val = self.reader.nextRawValue()

            key, = self._get_reader(raw_key).reads()
            val, = self._get_reader(raw_val).reads()
            return key, val

        def reads(self):
            return iter(self.read, None)
        __iter__ = reads


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
    except AttributeError:
        return False
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
    pts = inp._read()
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

    # do a little dance with this tempfile to avoid buffering issues
    with tempfile.NamedTemporaryFile(dir=tempdir, delete=False) as f:
        tempname = f.name

    try:
        index.save_index(tempname)
        index_size = os.path.getsize(tempname)

        out.file.write(struct.pack('>B', FLANN_CODE))
        out.file.write(struct.pack('>i',
            (1 + 4 + npy_size) + (4 + index_size)))
        # npy code, npy size int, npy, index size, index

        out.write(index.data)

        out.file.write(struct.pack('>i', index_size))
        with open(tempname, 'rb') as f:
            copyfileobj(f, out.file)
    finally:
        os.remove(tempname)

def register_write_flann(output_object):
    register_write_ndarray(output_object)
    output_object.register(cyflann.FLANNIndex, flann_to_typedbytes)


################################################################################

def register_read(inp):
    register_read_flann(inp)  # does read_ndarray itself

def register_write(out):
    register_write_flann(out)  # does write_ndarray itself
    register_np_writes(out)
