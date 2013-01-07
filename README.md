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

 * This code is written for Python 2.7 and 3.3.
   It may also work for 3.1 and/or 3.2; it is known not to work for 2.6,
   though adding support would not be overly difficult.
   It's also not actually tested on 3.x, but the source is written in a way
   that should be mostly compatible.

 * [numpy](numpy.scipy.org)

 * [SciPy](scipy.org)

 * [bottleneck](berkeleyanalytics.com/bottleneck/)

 * [FLANN](people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN),
   including the Python interface, is highly recommended for much
   faster nearest neighbor searches.

 * [h5py](code.google.com/p/h5py/) is optionally used in the
   command-line interface and in tests for input and output.

 * [scikit-learn](scikit-learn.org/) is used in the SDM code, for the
   LibSVM wrappers and some helper functions.

 * [progressbar](pypi.python.org/pypi/progressbar/)
   is optional in the CLI.

 * [nose](nose.readthedocs.org) for tests.

 * [argparse](http://pypi.python.org/pypi/argparse) is included in 2.7 and 3.2+,
   but is needed for older Pythons.
