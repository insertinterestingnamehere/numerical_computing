'''
Exercise solution: pasing a random vector between processes
>>> mpirun -n 2 python.exe passVector.py 6

*** Must be run with 2 processes
cmd line args: n, the length of the vector to pass between the nodes
'''
from sys import argv

import numpy as np
from mpi4py import MPI


# the length of the vector to pass:
n = int(argv[1])

#init process:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

ROOT = 0

# Abort if we don't have exactly 2 processes
if RANK == ROOT and SIZE != 2:
    print "ERROR: SIZE must be 2"
    COMM.Abort()



if RANK == ROOT:
    # Generate and send a random array to the other process
    vec = np.random.rand(n)
    print "Process {0}: Sent: vec={1}".format(RANK, vec)
    COMM.Send(vec, dest=1)
else:
    # Prepare and recieve the array from the root process
    vec = np.zeros(n)
    print "Process {0}: Before checking mailbox: vec={1}".format(RANK, vec)
    COMM.Recv(vec)
    print "Process {0}: Recieved: vec={1}".format(RANK, vec)
