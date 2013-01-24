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

 * [numpy](http://numpy.org)

 * [SciPy](http://scipy.org)

 * [bottleneck](http://berkeleyanalytics.com/bottleneck/) is used a little bit.

 * [FLANN](http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN),
   including the Python interface, is highly recommended for much
   faster nearest neighbor searches.

 * [h5py](http://code.google.com/p/h5py/) is optionally used in the
   command-line interface and in tests for input and output.

 * [scikit-learn](http://scikit-learn.org/) 0.13 or higher is used in the SDM
   code and in feature processing.

 * [vlfeat](http://vlfeat.org) is used in the image featurization code. We use
   a ctypes interface to libvl.(so|dylib); that should be on your path.

 * [scikit-image](http://scikit-image.org/) is used in the image featurization
   code for resizing. Additionally, you'll need either one of its plugins,
   opencv, or matplotlib and PIL installed to load images.

 * [progressbar](pypi.python.org/pypi/progressbar/) is optional in the CLI.

 * [nose](nose.readthedocs.org) for tests.

 * [argparse](http://pypi.python.org/pypi/argparse) is included in 2.7 and 3.2+,
   but is needed for older Pythons.
