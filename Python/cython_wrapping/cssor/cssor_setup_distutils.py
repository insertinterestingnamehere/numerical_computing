# Import needed setup functions.
from distutils.core import setup
from distutils.extension import Extension
# This is part of Cython's interface for distutils.
from Cython.Distutils import build_ext
# We still need to include the directory
# containing the NumPy headers.
from numpy import get_include
# We still need to run one command via command line.
from os import system

# Compile the .o file we will be accessing.
# this is independent of the process to build
# the Python extension module.
shared_obj = "gcc cssor.c -fPIC -c -o cssor.o"
print shared_obj
system(shared_obj)

# Tell Python how to compile the extension.
ext_modules = [Extension(
                         # Module name:
                         "cython_cssor",
                         # Cython source file:
                         ["cython_cssor.pyx"],
                         # Other compile arguments
                         # This flag doesn't do much
                         # this time, but this is
                         # where it would go.
                         extra_compile_args=["-fPIC", "-O3"],
                         # Extra files to link to:
                         extra_link_args=["cssor.o"])]

# Build the extension.
setup(name = 'cython_cssor',
      cmdclass = {'build_ext': build_ext},
      # Include the directory with the NumPy headers
      # when compiling.
      include_dirs = [get_include()],
      ext_modules = ext_modules)
