Installation
------------


As noted in ``README.rst``, it's much easier to use binary distributions of
``py-sdm`` through ``conda``. See the instructions there.

If you don't want to or can't do that, you can install like any other python
package, once you have the requirements set up properly.


Requirements
============


* `numpy <http://numpy.org>`_ and `scipy <http://scipy.org>`_ are not included
  in the pip requirements, to avoid over-eager reinstallation with
  ``pip install -U``.

* `FLANN <http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN>`_
  is used for fast nearest-neighbor searches,
  through the `cyflann <https://github.com/dougalsutherland/cyflann>`_ library.
  You need a more recent version than 1.8.4, which currently means you need to
  install from source (or use my binary on binstar).
  A version with OpenMP support is assumed (which means you need to compile with
  gcc, not clang or llvm-gcc).
  Note, though, that on OSX, versions of gcc prior to 4.6 will cause segfaults
  when FLANN uses OpenMP parallelism nested between py-sdm code and FLANN's.
  I use gcc-4.8 from Homebrew.

* `vlfeat-ctypes <https://github.com/dougalsutherland/vlfeat-ctypes>`_, a
  minimal ctypes interface to the `vlfeat <http://www.vlfeat.org>`_ computer
  vision algorithm library. This *is* installed by pip automatically, but
  make sure to run ``python -m vlfeat.download`` to download the library binary
  if you want to extract image features.

If you're using my conda channel but want to install py-sdm from source::

    conda install distribute numpy scipy cython flann cyflann scikit-learn \
                  scikit-image vlfeat-ctypes h5py nose progressbar


Actual installation
===================

You have several options. The most straightforward way to get a snapshot is to
just ``pip install`` the code (either into your system site-packages or
a `virtualenv <https://pypi.python.org/pypi/virtualenv>`_) via::

    pip install 'https://github.com/dougalsutherland/py-sdm/tarball/master#egg=sdm-0.1.0dev'

Since this package is still in early development, however, this somewhat
complicates the process of upgrading the code. It's easier to get upgrades if
you do a development install. If you have a checkout of the code, just do::

    python setup.py develop

Pip wil also check out the code for you into (by default) ``./src/py-sdm``::

    pip install -e 'git+https://github.com/dougalsutherland/py-sdm.git#egg=sdm-0.1.0dev'

In either case, if you issue a ``git pull`` in the source directory it'll update
the code.
