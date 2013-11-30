from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os import system

system("g++ -c Permutation.cpp Cycle.cpp Node.cpp")

ext_modules = [Extension("pypermutation",
                        sources=["pypermutation.pyx", "Permutation.pxd"],
                        language="c++",
                        extra_link_args=["Permutation.o","Cycle.o","Node.o"])]

setup(
    name="pypermutation",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules)
