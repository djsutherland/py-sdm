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
    msg = """py-sdm requires the python bindings for FLANN. These are available
from http://people.cs.ubc.ca/~mariusm/flann. Make sure to compile them with
OpenMP support (ie using gcc rather than clang).

(If you think you have FLANN installed, try running
    python PREFIX/share/flann/python/setup.py install
to get the bindings installed in your current python environment.)"""
    raise ImportError(msg)

try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

def cython_ext(name, **k):
    ret = []

    try:
        from Cython.Build import cythonize
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
        ret.extend(cythonize("sdm/{}.pyx".format(name)))

    ret.append(Extension("sdm.{}".format(name), ["sdm/{}.c".format(name)], **k))
    return ret

ext_modules = []
ext_modules.extend(cython_ext("_np_divs_cy",
    extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']))

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
)
# TODO: use "extras" to make some of the dependencies optional
#       but figure out how to make it work with distribute, etc...
