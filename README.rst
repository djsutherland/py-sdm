This is a Python implementation of nonparametric divergence estimators.

For an introduction to the method, see http://cs.cmu.edu/~dsutherl/sdm/

Homepage: https://github.com/dougalsutherland/py-sdm/

Code by Dougal J. Sutherland <dsutherl@cs.cmu.edu>
based partially on code by Liang Xiong <lxiong@cs.cmu.edu>.


Installation
------------

This code is written for Python 2.7, with 3.2+ compatability in mind (but not
tested). It is known not to work for 2.6, though adding support would not be
overly difficult; let me know if you want that.

It is also only tested on Unix-like operating systems (in particular, on OS X,
CentOS, and Ubuntu). All of the code except for the actual SVM wrappers
*should* work on Windows, but it's untested. The SVM wrappers *should* work
if you use n_proc=1; if you try to use multiprocessing there it will complain
and crash.

If you want to run with more than about a thousand objects, make sure that your
numpy and scipy are linked to a fast BLAS/LAPACK implementation like MKL, ACML,
or OpenBLAS.

The easiest way to accomplish that is to use a pre-packaged distribution. I use
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_. If you're affiliated
with an academic institution, you can get the MKL Optimizations add-on for free
that links numpy to Intel's fast MKL library. Anaconda (or EPD) also let you
avoid having to compile scipy (which takes a long time) and install non-python
libraries like hdf5. If so, ``conda install accelerate`` install them all.

It's also easiest to install py-sdm through binaries with the conda package
manager (part of Anaconda). There are currently only builds on 64-bit OSX and
64-bit Linux, with Python 2.7. To do so::

    conda install -c http://conda.binstar.org/dougal py-sdm

If you don't want to use binaries, there are various complications. See
``INSTALL.rst`` for details.


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
        --output-file feats_pca.cv.npz

for cross-validation. This will cache the calculated divergences in
``feats_pca.divs.h5``, and print out accuracy information as well as saving
predictions and some other info in ``feats_pca.cv.npz``.
This can take a long time, especially when doing divergences.

For regression, the command would look like::

    sdm cv --nu-svr --div-func renyi:.9 -K 5 --cv-folds 10 \
        --labels-name target_name
        feats_pca.h5 --div-cache-file feats_pca.divs.h5
        --output-file feats_pca.cv.npz

This uses ``--n-proc`` to specify the number of SVMs to run in parallel during
parameter tuning. During the projection phase (which happens in serial), an
MKL-linked numpy is likely to spawn many threads;
``OMP_NUM_THREADS`` will again control this.

Many more options are available via ``sdm cv --help``.

``sdm`` also supports predicting using a training / test set through
``sdm predict`` rather than ``sdm cv``, but there isn't currently code to
produce the input files it assumes. If this would be useful for you, let me
know and I'll write it....


Precomputing Divergences
========================

If you'd like to try several divergence functions (e.g. different values of
alpha or K), it's much more efficient to compute them all at once than to
let ``sdm`` do them all separately.

(This will hopefully no longer be true once ``sdm`` crossvalidates among
divergence functions and Ks:
`issue #12 <https://github.com/dougalsutherland/py-sdm/issues/12>`_.)

The ``estimate_divs`` command does this, using a command along the lines of::

    estimate_divs --div-funcs kl renyi:.8 renyi:.9 renyi:.99 -K 1 3 5 10 --
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
