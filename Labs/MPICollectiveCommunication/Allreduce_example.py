#Allreduce_example.py
from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
operand_buffer = np.array(float(RANK))
SIZE_buffer = np.zeros(1)

COMM.Allreduce(operand_buffer, SIZE_buffer, op=MPI.MAX)

SIZE = 1 + int(SIZE_buffer[0])
print "Rank {0}: The size is {1}.".format(RANK, SIZE)
