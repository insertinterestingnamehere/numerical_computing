'''
Takes a dot product in parallel.
Assumes n > SIZE
Does NOT assume that SIZE | n
CMD line argument: n, the length of the vector to dot with itself
'''

from mpi4py import MPI
import numpy as np
import collective_utilities as cu
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

local_n = n // SIZE
remainder = n % SIZE
if RANK < remainder:
    local_n += 1
local_x = np.zeros(local_n)
local_y = np.zeros(local_n)

COMM.Scatterv(x, local_x)
COMM.Scatterv(y, local_y)

local_dot_product = np.dot(local_x, local_y)
buf = np.array(local_dot_product)

product_buf = np.zeros(1) if RANK == ROOT else None
COMM.Reduce(buf, product_buf, MPI.SUM)

if RANK == ROOT:
    print "Parallel Dot Product: ", str(product_buf[0])
    print "Serial Dot Product: ", str(np.dot(x, y))
