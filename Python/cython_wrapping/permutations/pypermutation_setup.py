# Import needed setup functions
from distutils.core import setup
from distutils.extension import Extension
# This is part of Cython's interface for distutils.
from Cython.Distutils import build_ext
# We still need to run one command via command line.
from os import system

# Compile the .o files we will be accessing.
# This is independent of the process to build
# the Python extension module.
system("g++ -c Permutation.cpp Cycle.cpp Node.cpp")

# Tell Python how to compile the extension.
# Notice that we specify the language as C++ this time.
ext_modules = [Extension(
                        # Module name:
                        "pypermutation",
                        # Cython source files:
                        sources=["pypermutation.pyx", "Permutation.pxd"],
                        # Language:
                        language="c++",
                        # Other files needed for linking:
                        extra_link_args=["Permutation.o","Cycle.o","Node.o"])]

# Build the extension.
setup(
    name="pypermutation",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules)
