""" trapParallel_2.py
    Example usage:
        $ mpirun -n 10 python.exe trapParallel_2.py 0.0 1.0 10000
        With 10000 trapezoids, the estimate of the integral of x^2 from 0.0 to 1.0 is:
            0.333333335
    ***In this implementation, n must be divisble by the number of processes***
"""

from __future__ import division
from sys import argv
from mpi4py import MPI
import numpy as np


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def integrate_range(fxn, a, b, n):
    ''' Numerically integrates the function fxn by the trapezoid rule
        Integrates from a to b with n trapezoids
        '''
    # There are n trapezoids and therefore there are n+1 endpoints
    endpoints = np.linspace(a, b, n+1)

    integral = sum(fxn(x) for x in endpoints)
    integral -= (fxn(a) + fxn(b))/2
    integral *= (b - a)/n

    return integral

# An arbitrary test function to integrate
def function(x):
    return x**2

# Read the command line arguments
a = float(argv[1])
b = float(argv[2])
n = int(argv[3])


step_size = (b - a)/n
# local_n is the number of trapezoids each process will calculate
# ***Remember, in this implementation, n must be divisible by SIZE***
local_n = n / SIZE

# local_a and local_b are the start and end of this process' integration range
local_a = a + RANK*local_n*step_size
local_b = local_a + local_n*step_size

# mpi4py requires these to be numpy objects:
integral = np.zeros(1)
integral[0] = integrate_range(function, local_a, local_b, local_n)



# This has been the same as trapParallel_1.py up until this line. The rest is new:


total_buffer = np.zeros(1)

# The root node receives results with a collective "reduce"
COMM.Reduce(integral, total_buffer, op=MPI.SUM, root=0)
        
total = total_buffer[0]

# Now the root process prints the results:
if RANK == 0:
    print "With {0} trapezoids, the estimate of the integral of x^2 from {1} to {2} is: \n\t{3}".format(n, a, b, total)
