import numpy as np
from math import sqrt

# Problem 1
def arnoldi(A, b, Amul, k, tol=1E-8):
    """Perform 'k' steps of the Arnoldi Iteration for
    sparse array 'A' and starting point 'b'with multiplicaiton
    function 'Amul'. Stop if the projection of a vector
    orthogonal to the current subspace has norm less than 'tol'."""
    # Some initialization steps.
    Q = np.empty((A.shape[0], k+1), order='F')
    H = np.zeros((k+1, k), order='F')
    ritz_vals = []
    # Set q_0 equal to b.
    Q[:,0] = b
    # Normalize q_0.
    Q[:,0] /= sqrt(np.inner(b, b))
    # Perform actual iteration.
    for j in xrange(k):
        # Compute A.dot(q_j).
        Q[:,j+1] = Amul(Q[:,j])
        # Modified Graham Schmidt
        for i in xrange(j+1):
            # Set values of $H$
            H[i,j] = np.inner(Q[:,i], Q[:,j+1])
            Q[:,j+1] -= H[i,j] * Q[:,i]
        # Set subdiagonel element of H
        H[j+1,j] = sqrt(np.inner(Q[:,j+1], Q[:,j+1]))
        # Stop if ||q_{j+1}|| is too small.
        if abs(H[j+1, j]) < tol:
            return np.asfortranarray(H[:j+1,:j]), Q[:,j]
        # Normalize q_{j+1}
        Q[:,j+1] /= H[j+1, j]
    return H, Q
