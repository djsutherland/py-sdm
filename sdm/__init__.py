from . import utils
from . import mp_utils

from . import features
from .features import Features

from . import np_divs
from .np_divs import estimate_divs

from . import sdm
from .sdm import SDC, NuSDC, SDR, NuSDR, OneClassSDM

try:
    from numpy.testing import nosetester
    test = nosetester.NoseTester().test
    del nosetester
except ImportError:
    pass
