This is a pure-Python implementation of the nonparametric divergence estimators
described by

Barnabas Poczos, Liang Xiong, Jeff Schneider (2011).
*Nonparametric divergence estimation with applications to machine learning on distributions.*
Uncertainty in Artificial Intelligence.
http://autonlab.org/autonweb/20287.html

and also their use in support vector machines, as described by

Dougal J. Sutherland, Liang Xiong, Barnabas Poczos, Jeff Schneider (2012).
*Kernels on Sample Sets via Nonparametric Divergence Estimates.*
http://arxiv.org/abs/1202.0302

Code by Dougal J. Sutherland <dsutherl@cs.cmu.edu>
based partially on code by Liang Xiong <lxiong@cs.cmu.edu>.



Installation
------------


Requirements
============

This code is written for Python 2.7, with 3.2+ compatability in mind (but not
tested). It is known not to work for 2.6, though adding support would not be
overly difficult; let me know if you want that.

It is also only tested on Unix-like operating systems (in particular, on OS X,
CentOS, and Ubuntu). All of the code except for the actual SVM wrappers
*should* work on Windows, but it's untested. The SVM wrappers *should* work
if you use n_proc=1; if you try to use multiprocessing there it will complain
and crash.

The `Enthought Python Distribution <http://www.enthought.com/epd>`_, which
has an academic license, includes numpy, scipy, h5py, and PIL -- the hardest to
install of the dependencies. It also includes scikit-image and scikit-learn, but
we need newer versions than it ships with.

* `numpy <http://numpy.org>`_ and `scipy <http://scipy.org>`_ are not included
  in the pip requirements, to avoid over-eager reinstallation with
  ``pip install -U``. If you're going to run on largeish datasets, the PCA and
  PSD projection stages will be faster if your numpy is linked to a fast
  BLAS/LAPACK like MKL (true of EPD).

* `FLANN <http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN>`_,
  including the Python interface, is highly recommended for much faster nearest
  neighbor searches. This isn't in pip but is in homebrew, ubuntu's apt, etc.

* `vlfeat-ctypes <https://github.com/dougalsutherland/vlfeat-ctypes>`_, a
  minimal ctypes interface to the `vlfeat <http://www.vlfeat.org>`_ computer
  vision algorithm library. This *is* installed by pip automatically, but
  make sure to run ``python -m vlfeat.download`` to download the library binary.


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

If you want to use the image feature code you need to either install
``libvl.so`` (or similar for your platform)
or issue ``python -m vlfeat.download`` to install it in your site-packages.
(Adding ``-h`` to that shows how to avoid re-downloading the binary distribution
if you already have it.)



Quick Start Guide for Images
----------------------------

This shows you the basics of how to do classification or regression on images.


Data Format
===========

If you're doing classification, it's easiest if your images are in a single
directory containing one directory per class, and images for that class in the
directory: ``root-dir/class-name/image-name.jpg``

If you're doing regression, it's easiest to have your images all in a single
directory, and a CSV file $target_name.csv with labels of the form::

    image1.jpg,2.4

for each image in the directory (no header).


Extracting Features
===================

This step extracts SIFT features for a collection of images.

The basic command is something like::

    extract_image_features --root-dir path-to-root-dir --color hsv feats_raw.h5

for classification, or::

    extract_image_features --dirs path-to-root-dir --color hsv feats_raw.h5

for regression.

This by default spawns one process per core to extract features (each of which
uses only one thread); this can be controlled with the ``--n-proc`` argument.

You're likely to want to use the ``--resize`` option if your images are large
and/or of widely varying sizes. We typically resize them to be about 100px wide
or so.

See ``--help`` for more options.


Post-Processing Features
========================

This step handles "blanks," does dimensionality reduction via PCA, adds
spatial information, and standardizes features.

The basic command is::

    proc_image_features --pca-varfrac 0.7 feats_raw.h5 feats_pca.h5

This by default does a dense PCA; if you have a lot of images and/or the images
are large, it'll take a lot of memory.
You can reduce memory requirements a lot by replacing the ``--pca-varfrac 0.7``
with something like ``--pca-k 50 --pca-random``, which will do a randomized SVD
to reduce dimensionality to 50; you have to specify a specific dimension rather
than a percent of variance, though.

If you have a numpy linked to MKL or other fancy blas libraries, it will
probably try to eat all your cores during the PCA; the ``OMP_NUM_THREADS``
environment variable can limit that.

Again, other options available via ``--help``.


Classifying/Regressing
======================

Once you have this, to calculate divergences and run the SVMs in one step you
can use a command like::

    sdm cv --div-func renyi:.9 -K 5 --cv-folds 10 \
        feats_pca.h5 --div-cache-file feats_pca.divs.h5 \
        --output-file feats_pca.cv.mat

for cross-validation. This will cache the calculated divergences in
``feats_pca.divs.h5``, and print out accuracy information as well as saving
predictions and some other info in ``feats_pca.cv.mat``.
This can take a long time, especially when doing divergences.

For regression, the command would look like::

    sdm cv --nu-svr --div-func renyi:.9 -K 5 --cv-folds 10 \
        --labels-name target_name
        feats_pca.h5 --div-cache-file feats_pca.divs.h5
        --output-file feats_pca.cv.mat

This uses ``--n-proc`` to specify the number of SVMs to run in parallel during
parameter tuning. During the projection phase (which happens in serial), an
MKL-linked numpy is likely to spawn many threads;
``OMP_NUM_THREADS`` will again control this.

Many more options are available via ``sdm cv --help``.

``sdm`` also supports predicting using a training / test set through
``sdm predict`` rather than ``sdm cv``, but there isn't currently code to
produce the input files it assumes.


Precomputing Divergences
========================

If you'd like to try several divergence functions (e.g. different values of
alpha or K), it's much more efficient to compute them all at once than to
let ``sdm`` do them all separately.

(This will hopefully no longer be true once ``sdm`` crossvalidates among
divergence functions and Ks:
`issue #12 <https://github.com/dougalsutherland/py-sdm/issues/12>`_.)

The ``extract_divs`` command does this, using a command along the lines of::

    extract_divs --div-funcs kl renyi:.8 renyi:.9 renyi:.99 -K 1 3 5 10 --
        feats_pca.h5 feats_pca.divs.h5

(where the ``--`` indicates that the ``-K`` arguments are done and it's time for
positional args.)



Quick Start Guide For General Features
--------------------------------------

If you don't want to use the image feature extraction code above, you have two
main options for using SDMs.


Making Compatible Files
=======================

One option is to make an hdf5 file compatible with the output of
``extract_image_features`` and ``proc_image_features``, e.g. with ``h5py``.
The structure that you want to make is::

    /cat1          # the name of a category
      /bag1        # the name of each data sample
        /features  # a row-instance feature matrix
        /label-1   # a scalar dataset with the value of label-1
        /label-2   # scalar dataset with a second label type
      /bag2
        ...
    /cat2
      ...

Some notes:

* All of the names except ``features`` can be replaced with whatever you like.
* If you have a single "natural" classification label, it can be convenient to
  use that for the category, but you can put them all in the same category if
  you like.
* The features matrices can have any number of rows but must have the same
  numbers of columns.
* Different bags need not have the same labels available, unless you want to use
  them for training / cross-validating in ``sdm``. Each bag can have any number
  of labels.

Alternatively, you can use the "per-bag" format, where you make a ``.npz``
file (with ``np.savez``) at ``root-path/cat-name/bag-name.npz`` with a
``features`` matrix and any labels (as above).

Depending on the nature of your features, you may want to run PCA on them,
standardize the dimensions, or perform other normalizations. You can do PCA and
standardization with ``proc_image_features``, as long as you make sure to pass
``--blank-handler none --no-add-x --no-add-y`` so it doesn't try to do image-
specific stuff.

You can then use ``sdm`` as above.


Using the API
=============

You can also use the API directly. The following shows basic usage in the
situation where test data is not available at training time::

    import sdm

    # train_features is a list of row-instance data matrices
    # train_labels is a numpy vector of integer categories

    # PCA and standardize the features
    train_feats = sdm.Features(train_features)
    pca = train_feats.pca(varfrac=0.7, ret_pca=True, inplace=True)
    scaler = train_feats.standardize(ret_scaler=True, inplace=True)

    clf = sdm.SDC()
    clf.fit(train_feats, train_labels)
    # ^ gets divergences and does parameter tuning. See the docstrings for
    # more information about options, divergence caches, etc. Caching
    # divergences is highly recommended.

    # get test_features: another list of row-instance data matrices
    # and then process them consistently with the training samples
    test_feats = sdm.Features(test_features, default_category='test')
    test_feats.pca(pca=pca, inplace=True)
    test_feats.normalize(scaler=scaler, inplace=True)

    # get test predictions
    preds = clf.predict(test_feats)

    accuracy = np.mean(preds == test_labels)

To do regression, use ``clf = sdm.NuSDR()`` and a real-valued train_labels;
the rest of the usage is the same.

If you're running on a nontrivial amount of data, it may be nice to pass
``status_fn=True`` and ``progressbar=True`` to the constructor to get status
information out along the way (like in the CLI).

If test data is available at training time, it's preferable to use
``.transduct()`` instead. There's also a ``.crossvalidate()`` method.
