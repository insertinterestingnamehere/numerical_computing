# Import needed setup functions.
from distutils.core import setup
from distutils.extension import Extension
# This is part of Cython's interface for distutils.
from Cython.Distutils import build_ext
# We still need to include the directory
# containing the NumPy headers.
from numpy import get_include
# We still need to run one command via command line
from os import system

# Compile the .o file we will be accessing.
# this is independent of the process to build
# the Python extension module.
system("gcc ctridiag.c -c -o ctridiag.o")

# Tell Python how to compile the extension.
ext_modules = [Extension("cytridiag",
                         # Module name:
                         ["cytridiag.pyx"],
                         # Other compile arguments
                         # This flag isn't necessary
                         # this time, but this is
                         # where it would go.
                         extra_compile_args=["-O3"],
                         # Extra files to link to.
                         extra_link_args=["ctridiag.o"])]

# Build the extension.
setup(name = 'cytridiag',
      cmdclass = {'build_ext': build_ext},
      include_dirs = [get_include()],
      ext_modules = ext_modules)
