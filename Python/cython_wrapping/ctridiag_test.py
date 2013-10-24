import numpy as np
from numpy.random import rand
# Import the new module.
from cython_ctridiag import cytridiag as ct

a, b, c, x = rand(9), rand(10), rand(9), rand(10)
# Call the function.
ct(a, b, c, x)
# The printed values should be somewhat different
# than the usual output from rand().
# There should probably be some negative values
# and definitely no extemely large or extremely
# small values.
print x
