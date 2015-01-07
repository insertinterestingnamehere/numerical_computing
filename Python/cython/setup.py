from distutils.core import setup
from Cython.Build import cythonize

setup(name="hello", ext_modules=cythonize('hello.pyx'))
