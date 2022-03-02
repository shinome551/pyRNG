from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("rng", sources=["rng.pyx"],
    include_dirs=['.', get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
setup(name="rng", ext_modules=cythonize([ext]))