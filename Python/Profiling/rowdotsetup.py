from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(name = "rowdot", ext_modules = cythonize('rowdot.pyx'), include_dirs=[numpy.get_include()], )
