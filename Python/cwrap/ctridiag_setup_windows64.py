from os import system
from numpy import get_include
from distutils.sysconfig import get_python_inc
from sys import prefix

ic = " -I" + get_python_inc()
npy = " -I" + get_include()
lb = " \"" + prefix + "/libs/libpython27.a\""
o = " \"ctridiag.o\""

flags = ic + npy + lb + o + " -D MS_WIN64"

system("gcc ctridiag.c -c -o ctridiag.o")
system("cython -a cytridiag.pyx --embed")
system("gcc -c cytridiag.c" + flags)
system("gcc -shared cytridiag.o -o cytridiag.pyd" + flags)
