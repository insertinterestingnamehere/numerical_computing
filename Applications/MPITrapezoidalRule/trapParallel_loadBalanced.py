'''
Computes the integral of x**2 through parallel computing
Example: mpirun -n 4 python.exe ex4_6_4.py 0.0 1.0 10000

This one is load-balancing and will work when num_threads does not divide evenly into n
'''

from __future__ import division


from sys import argv
import numpy as np
from mpi4py import MPI
import collective_utilities as cu


COMM = MPI.COMM_WORLD

RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
ROOT = 0

a, b, n = argv[1:4]
a = float(a)
b = float(b)
n = int(n)

#an arbitrarily defined function to integrate:
def g(x):
    return x**2


def integrate_range(a, b, n, f=g):
    ''' Serially integrates from a to b with n trapezoids
        (We'll only call this on a small part of the full domain for each process) 
    '''
    integral = -(f(a) + f(b))/2.0

    # n+1 endpoints, but n trapezoids
    integral += sum(f(x) for x in np.linspace(a, b, n+1))
    
    return integral * (b-a)/n


def ordered_balance(comm, num_elements):
    rank = comm.Get_rank()
    size = comm.Get_size()
    my_share = num_elements // size
    remainder = num_elements % size

    if rank < remainder:
        my_share += 1
        displacement = rank*my_share
    else:
        displacement = remainder*(my_share+1) + (rank-remainder)*my_share
    return my_share, displacement


def get_local_params(comm, a, b, n):
    ''' Calculates local_n, the number of trapezoid that are allocated to this process
        This is load_balancing- the difference in local_n between any two nodes should be at most 1

        Example logic:
            a, b, n = 0, 1, 10
            num_nodes = 4
            
            local_n indexed by process = [2, 2, 2, 2]
            num_orphan_traps = 10 - 2*4 = 2
            local_n indexed by process = [3, 3, 2, 2]
    '''
    # h is the step size, n is the total number of trapezoids
    h = (b-a)/n

    local_n, displacement = ordered_balance(comm, n)
    local_a = a + displacement*h
    local_b = local_a + local_n*h

    return local_a, local_b, local_n


local_params = get_local_params(COMM, a, b, n)
# cu.par_print(COMM, 'Process ' + str(RANK) + ' has local_params = ' + str(local_params) ) #debug


#mpi4py requires these to be numpy objects
integral = np.zeros(1)
integral[0] = integrate_range(*local_params)
COMM.Send(integral) # send to the root node, even if I am ROOT

# the root node now compiles the results
if RANK == ROOT:
    total = 0
    buf = np.zeros(1)
    for i in xrange(SIZE):
        COMM.Recv(buf, MPI.ANY_SOURCE)
        total += buf[0]
    
    print "With n = {0} trapezoids, the integral of f from {1} to {2} is: {3}.".format(n, a, b, total)
    print "Parallel Computation:", total
    print "Serial Computation:  ", integrate_range(a, b, n)
