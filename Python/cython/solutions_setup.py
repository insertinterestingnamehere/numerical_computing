from distutils.core import setup
from Cython.Build import cythonize

setup(name="solutions", ext_modules=cythonize('solutions.pyx'))
