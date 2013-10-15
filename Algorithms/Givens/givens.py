import numpy as np
from math import sqrt

def givens(A, tol=1E-15):
    # Make R a copy of A and Q an
    # identity array of the appropriate size.
    R = np.array(A, order="C")
    Q = np.eye(A.shape[0])
    # Make an empty 2x2 array G that will
    # be used to apply the Givens rotations.
    G = np.empty((2,2))
    # For each column:
    for j in xrange(A.shape[1]-1):
        # For each row below the main diagonal
        # (starting at the bottom of the column):
        for i in xrange(A.shape[0]-1, j, -1):
            # If the leading entry of this row is
            # not zero (i.e. if its absolute value
            # is within a given tolerance):
            if tol <= abs(R[i,j]):
                # Compute c and s using the entry
                # in the current row and column
                # and the entry immediately above it.
                c = R[i-1,j]
                s = - R[i,j]
                n = sqrt(c**2 + s**2)
                c /= n
                s /= n
                # Use c and s to construct the matrix G.
                G[0,0] = c
                G[1,1] = c
                G[0,1] = - s
                G[1,0] = s
                # Get a slice of $R$ of the current row
                # and the row above it that includes the
                # columns from the current column onward.
                # Multiply it in place by $G$ to zero out
                # the leading nonzero entry of the current row.
                R[i-1:i+1,j:] = G.dot(R[i-1:i+1,j:])
                # Get a slice of $Q$ of the current row and
                # the row above it and apply $G$ to it as well.
                # Here we use the fancy slicing to avoid extra
                # computations in creating Q.
                # The column slicing is not in the lab.
                Q[i-1:i+1,min(i-1-j,0):] = G.dot(Q[i-1:i+1,min(i-1-j,0):])
    # Return Q^T and R.
    return Q.T, R

def givens_hess(A, tol=1E-15):
    # Make R a copy of A and Q an
    # identity array of the appropriate size.
    R = np.array(A, order="C")
    Q = np.eye(A.shape[0])
    # Make an empty 2x2 array G that will
    # be used to apply the Givens rotations.
    G = np.empty((2,2))
    # Iterate along first subdiagonal.
    # Run index along columns.
    for j in xrange(A.shape[0]-1):
        # If the leading entry of this row is
        # not zero (i.e. if its absolute value
        # is within a given tolerance):
        if tol <= abs(R[i,j]):
            # Compute c and s using the entry
            # in the current row and column
            # and the entry immediately above it.
            c = R[j,j]
            s = - R[j+1,j]
            n = sqrt(c**2 + s**2)
            c /= n
            s /= n
            # Use c and s to construct the matrix G.
            G[0,0] = c
            G[1,1] = c
            G[0,1] = -s
            G[1,0] = s
            # Apply rotation to proper portion of R.
            R[j:j+2,j:] = G.dot(R[j:j+2,j:])
            # Apply rotation to proper portion of Q
            Q[j:j+2,:j+2] = G.dot(Q[j:j+2,:j+2])
    # Return Q^T and R.
    return Q.T, R
