'''
Takes a dot product in parallel.
Example usage:
    $ mpirun -n 4 python.exe dot.py 1000
Assumes n is divisible by SIZE

command line arguments: n, the length of the vector to dot with itself
'''
from __future__ import division

from mpi4py import MPI
import numpy as np
from sys import argv


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

ROOT = 0

n = int(argv[1])


if RANK == ROOT:
    x = np.linspace(0, 100, n)
    y = np.linspace(20, 300, n)
else:
    x, y = None, None

# Prepare variables
local_n = n // SIZE
if n % SIZE != 0:
    print "The number of processors must evenly divide n."
    COMM.Abort()

local_x = np.zeros(local_n)
local_y = np.zeros(local_n)

COMM.Scatter(x, local_x)
COMM.Scatter(y, local_y)

local_dot_product = np.dot(local_x, local_y)
buf = np.array(local_dot_product)

result_buf = np.zeros(1) if RANK == ROOT else None
COMM.Reduce(buf, result_buf, MPI.SUM)

if RANK == ROOT:
    print("Parallel Dot Product: " + str(result_buf[0]))
    print("Serial Dot Product: " + str(np.dot(x, y)))
