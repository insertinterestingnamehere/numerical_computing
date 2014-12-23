#matrix_scatter_example
from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
local_a = np.zeros(3)
COMM.Scatter(A, local_a)
print "Process {0} has {1}.".format(RANK, local_a)

