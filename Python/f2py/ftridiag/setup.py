from numpy.distutils.core import setup, Extension
ext = Extension('ftridiag',
                sources=['ftridiag.f90'],
                extra_compile_args=['-O3'])
setup(ext_modules=[ext])
