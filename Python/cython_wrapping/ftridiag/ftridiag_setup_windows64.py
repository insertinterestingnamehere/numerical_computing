from os import system
from numpy import get_include
from distutils.sysconfig import get_python_inc
from sys import prefix

ic = " -I" + get_python_inc()
npy = " -I" + get_include()
lb = " \"" + prefix + "/libs/libpython27.a\""
o = " \"ftridiag.o\""

flags = ic + npy + lb + o + " -D MS_WIN64"

system("gfortran ftridiag.f90 -c -o ftridiag.o")
system("cython -a cython_ftridiag.pyx --embed")
system("gcc -c cython_ftridiag.c" + flags)
system("gcc -shared cython_ftridiag.o -o cython_ftridiag.pyd" + flags)
