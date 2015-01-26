from distutils.core import setup
from Cython.Build import cythonize

setup(name="cymodule", ext_modules=cythonize('cymodule.pyx'))
