try:
    import numpy
    import scipy
except ImportError:
    raise ImportError("py-sdm requires numpy and scipy to be installed")
    # Don't do this in the setup() requirements, because otherwise pip and
    # friends get too eager about updating numpy/scipy.

try:
    import pyflann
except ImportError:
    import warnings
    warnings.warn("The python FLANN bindings don't seem to be installed. "
                  "These are highly recommended.", ImportWarning)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='py-sdm',
    version='0.1.0dev',
    author='Dougal J. Sutherland',
    author_email='dougal@gmail.com',
    packages=['sdm'],
    url='https://github.com/dougalsutherland/py-sdm',
    description='An implementation of nonparametric divergence estimators and '
                'their use in SVMs.',
    long_description=open('README.rst').read(),
    license='LICENSE.txt',
    install_requires=[
        'h5py',
        'progressbar',
        'scikit-learn >= 0.13',
        'scikit-image >= 0.6',
        'bottleneck',
        'nose',
        'vlfeat-ctypes',
    ],
    dependency_links=[
        'https://github.com/dougalsutherland/vlfeat-ctypes/tarball/master#egg=vlfeat-ctypes-0.1.0dev'
    ],
    entry_points={
        'console_scripts': [
            'extract_image_features = sdm.extract_image_features:main',
            'proc_image_features = sdm.proc_image_features:main',
            'estimate_divs = sdm.np_divs:main',
            'sdm = sdm.sdm:main',
        ],
    },
)
# TODO: use "extras" to make some of the dependencies optional
#       but figure out how to make it work with distribute, etc...
