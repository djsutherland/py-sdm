try:
    import numpy
    import scipy
except ImportError:
    raise ImportError("py-sdm requires numpy and scipy to be installed")
    # Don't do this in the setup() requirements, because otherwise pip and
    # friends get too eager about updating numpy/scipy.

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension

setup_args = {}

# Handle the _np_divs_cy cython extension. It has enough specific handling that
# it's not worth breaking out into a function.
name = "_np_divs_cy"
try:
    from Cython.Distutils import build_ext
except ImportError:
    import os
    try:
        pyx_time = os.path.getmtime('sdm/{}.pyx'.format(name))
        c_time = os.path.getmtime('sdm/{}.c'.format(name))
        if pyx_time >= c_time:
            raise ValueError
    except (OSError, ValueError):
        msg = "{} extension needs to be compiled but cython isn't available"
        raise ImportError(msg.format(name))
    else:
        source_file = "sdm/{}.c".format(name)
else:
    try:
        import cyflann
    except ImportError:
        msg = \
"""The Cython extension requires cyflann to be installed before compilation.
Install cyflann (e.g. `pip install cyflann`), and then try again."""
        raise ImportError(msg)

    source_file = "sdm/{}.pyx".format(name)
    setup_args['cmdclass'] = {'build_ext': build_ext}

ext_modules = [
    Extension("sdm.{}".format(name), [source_file],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp', '-lflann'])
]


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
        'nose',
        'cyflann',

        # only for image feat extraction; should be "extras"
        'scikit-image >= 0.6',
        'vlfeat-ctypes',
    ],
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'extract_image_features = sdm.extract_image_features:main',
            'proc_image_features = sdm.proc_image_features:main',
            'estimate_divs = sdm.np_divs:main',
            'sdm = sdm.sdm:main',
        ],
    },
    **setup_args
)
# TODO: use "extras" to make some of the dependencies optional
#       but figure out how to make it work with distribute, etc...
