from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
from os import system

system("gfortran ftridiag.f90 -c -o ftridiag.o")

ext_modules = [Extension("cython_ftridiag", ["cython_ftridiag.pyx"],
               extra_compile_args=["-O3"],
               extra_link_args=["ftridiag.o"])]

setup(name = 'cython_ftridiag',
    cmdclass = {'build_ext': build_ext},
    include_dirs = [get_include()],
    ext_modules = ext_modules)
