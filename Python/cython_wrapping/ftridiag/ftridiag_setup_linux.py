from os import system
from numpy import get_include
from distutils.sysconfig import get_python_inc

ic = " -I" + get_python_inc()
npy = " -I" + get_include()
o = " ftridiag.o"

flags = ic + npy + o + " -fPIC"

system("gfortran ftridiag.f90 -c -o ftridiag.o -fPIC")
system("cython -a cytridiag.pyx --embed")
system("gcc -c cytridiag.c" + flags)
system("gcc -shared cytridiag.o -o cytridiag.so" + flags)
