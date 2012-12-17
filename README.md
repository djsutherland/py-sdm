This is a pure-Python implementation of the nonparametric divergence estimators 
described by

Barnabas Poczos, Liang Xiong, Jeff Schneider (2011).
_Nonparametric divergence estimation with applications to machine learning on distributions._
Uncertainty in Artificial Intelligence.
http://autonlab.org/autonweb/20287.html

Code by
 * Liang Xiong (lxiong@cs.cmu.edu)
 * Dougal J. Sutherland (dsutherl@cs.cmu.edu)


Requirements
------------

 * This code is written for Python 2.7 and 3.3.
   It may also work for 3.1 and/or 3.2; it is known not to work for 2.6,
   though adding support would not be overly difficult.

 * [numpy](numpy.scipy.org)

 * [SciPy](scipy.org)

 * [FLANN](http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN),
   including the Python interface, is highly recommended for much
   faster nearest neighbor searches.

 * [h5py](http://code.google.com/p/h5py/) is optionally used in the
   command-line interface for input and output.

 * [progressbar](http://pypi.python.org/pypi/progressbar/)
   is optional in the CLI.

 * [argparse](http://pypi.python.org/pypi/argparse) is included in 2.7 and 3.2+,
   but is needed for older Pythons.
