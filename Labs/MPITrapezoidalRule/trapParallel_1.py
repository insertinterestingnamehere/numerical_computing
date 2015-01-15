""" trapParallel_1.py
    Example usage:
        $ mpirun -n 10 python.exe trapParallel_1.py 0.0 1.0 10000
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


if RANK != 0:
    # Send the result to the root node; the destination parameter defaults to 0
    COMM.Send(integral)
else:
    # The root node now compiles and prints the results
    total = integral[0]
    communication_buffer = np.zeros(1)
    for _ in xrange(SIZE-1):
        COMM.Recv(communication_buffer, MPI.ANY_SOURCE)
        total += communication_buffer[0]
    
    print "With {0} trapezoids, the estimate of the integral of x^2 from {1} to {2} is: \n\t{3}".format(n, a, b, total)
