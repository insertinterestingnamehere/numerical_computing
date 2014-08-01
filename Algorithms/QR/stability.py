import numpy as np
from numpy.random import rand
from scipy import linalg as la

Q, X = la.qr(rand(500,500)) # create a random orthonormal matrix:
R = np.triu(rand(500,500)) # create a random upper triangular matrix
A = np.dot(Q,R) # Q and R are the exact QR decomposition of A
Q1, R1 = la.qr(A) # compute QR decomposition of A