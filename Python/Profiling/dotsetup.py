from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(name = "dot", ext_modules = cythonize('dot.pyx'), include_dirs=[numpy.get_include()], )
