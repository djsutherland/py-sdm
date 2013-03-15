This is a pure-Python implementation of the nonparametric divergence estimators
described by

Barnabas Poczos, Liang Xiong, Jeff Schneider (2011).
_Nonparametric divergence estimation with applications to machine learning on distributions._
Uncertainty in Artificial Intelligence.
http://autonlab.org/autonweb/20287.html

and also their use in support vector machines, as described by

Dougal J. Sutherland, Liang Xiong, Barnabas Poczos, Jeff Schneider (2012).
_Kernels on Sample Sets via Nonparametric Divergence Estimates._
http://arxiv.org/abs/1202.0302

Code by Dougal J. Sutherland <dsutherl@cs.cmu.edu>
based on code by Liang Xiong <lxiong@cs.cmu.edu>.


Requirements
------------

 * This code is written for Python 2.7 with 3.2+ compatability in mind.
   It is known not to work for 2.6, though adding support would not be overly difficult.
   The [Enthought Python Distribution](http://www.enthought.com/epd), which
   has an academic license, includes numpy, scipy, h5py, and PIL -- the hardest to
   install of the dependencies. It also includes scikit-image and scikit-learn, but we
   need newer versions than it ships with.

 * [numpy](http://numpy.org) - if you're going to run on largeish datasets,
   the PCA and PSD projection stages will be faster if your numpy is linked to a
   fast BLAS/LAPACK like MKL (true of EPD).

 * [SciPy](http://scipy.org)

 * [bottleneck](http://berkeleyanalytics.com/bottleneck/) is used a little bit.

 * [FLANN](http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN),
   including the Python interface, is highly recommended for much
   faster nearest neighbor searches. This isn't in pip but is in homebrew,
   ubuntu's apt, etc.

 * [h5py](http://code.google.com/p/h5py/) is optionally used in the
   command-line interface and in tests for input and output.

 * [scikit-learn](http://scikit-learn.org/) 0.13 or higher is used in the SDM
   code and in feature processing.

 * [vlfeat](http://vlfeat.org) is used in the image featurization code. We use
   a ctypes interface to libvl.(so|dylib); that should be on your path.
   (Get it from http://www.vlfeat.org/download/vlfeat-0.9.16-bin.tar.gz,
   and just extract e.g. vlfeat-0.9.16/bin/glnxa64/libvl.so.)

 * [scikit-image](http://scikit-image.org/) is used in the image featurization
   code for resizing. Additionally, you'll need either one of its plugins,
   opencv, or matplotlib and PIL installed to load images.

 * [progressbar](pypi.python.org/pypi/progressbar/) is optional in the CLI.

 * [nose](nose.readthedocs.org) for tests.



Quick Start Guide
-----------------

This shows you the basics of how to do classification or regression on images.

Data Format
===========

If you're doing classification, it's easiest if your images are in a single
directory containing one directory per class, and images for that class in the
directory: `root-dir/class-name/image-name.jpg`

If you're doing regression, it's easiest to have your images all in a single
directory, and a CSV file $target_name.csv with labels of the form

    image1.jpg,2.4

for each image in the directory (no header).



Extracting Features
===================

This step extracts SIFT features for a collection of images.

The basic command is something like

    ./extract_features.py --root-dir path-to-root-dir --color hsv feats_raw.h5

for classification, or

    ./extract_features.py --dirs path-to-root-dir --color hsv feats_raw.h5

for regression.

This by default spawns one process per core to extract features (each of which
uses only one thread); this can be controlled with the `--n-proc` argument.

You're likely to want to use the --resize option if you're images are large
and/or of widely varying sizes.

See `--help` for more options.



Post-Processing Features
========================

This step handles "blanks", does dimensionality reduction via PCA, adds
spatial information, and standardizes features.

The basic command is

    ./proc_features.py --pca-varfrac 0.7 feats_raw.h5 feats_pca.h5

This by default does a dense PCA; if you have a lot of images and/or the images
are large, it'll take a lot of memory.
You can reduce memory requirements a lot by replacing the `--pca-varfrac 0.7`
with something like `--pca-k 50 --pca-random`, which will do a randomized SVD
to reduce dimensionality to 50; you have to specify a specific dimension rather
than a percent of variance, though.

If you have a numpy linked to MKL or other fancy blas libraries, it will
probably try to eat all your cores during the PCA; the `OMP_NUM_THREADS`
environment variable can limit that.

Again, other options available via `--help`.



Classifying/Regressing
======================

Once you have this, to calculate divergences and run the SVMs in one step you
can use a command like

    ./sdm.py cv --div-func renyi:.9 -K 5 --cv-folds 10 \
        feats_pca.h5 --div-cache-file feats_pca.divs.h5 \
        --output-file feats_pca.cv.mat

for cross-validation. This will cache the calculated divergences in
`feats_pca.divs.h5`, and print out accuracy information as well as saving
predictions and some other info in `feats_pca.cv.mat`.
This can take a long time, especially when doing divergences.

For regression, the command would look like

    ./sdm.py cv --nu-svr --div-func renyi:.9 -K 5 --cv-folds 10 \
        --labels-name target_name
        feats_pca.h5 --div-cache-file feats_pca.divs.h5
        --output-file feats_pca.cv.mat

This uses `--n-proc` to specify the number of SVMs to run in parallel during parameter tuning. During the projection phase (which happens in serial), an MKL-linked numpy is likely to spawn many threads; `OMP_NUM_THREADS` will again control this.

Many more options are available via `sdm.py cv --help`.

`sdm.py` also supports predicting using a training / test set through
`sdm.py predict` rather than `sdm.py cv`, but there isn't currently code to produce the input files it assumes.



Precomputing Divergences
========================

If you'd like to try several divergence functions (e.g. different values of
alpha or K), it's much more efficient to compute them all at once than to
let `sdm.py` do them all separately.

(This will hopefully no longer be true once `sdm.py` crossvalidates among
divergence functions: issue #12.)

The `get_divs.py` command does this, using a command along the lines of

    ./get_divs.py --div-funcs renyi:.8,.9,.99 -K 1 3 5 10 --
        feats_pca.h5 feats_pca.divs.h5

(where the `--` indicates that the `-K` arguments are done and it's time for
(positional args.)
