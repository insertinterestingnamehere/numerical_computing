import numpy as np
from cmath import sqrt

# Problem 1
def arnoldi(b, Amul, k, tol=1E-8):
    """Perform 'k' steps of the Arnoldi Iteration for
    sparse array 'A' and starting point 'b'with multiplicaiton
    function 'Amul'. Stop if the projection of a vector
    orthogonal to the current subspace has norm less than 'tol'."""
    # Some initialization steps.
    # Initialize to complex arrays to avoid some errors.
    Q = np.empty((b.size, k+1), order='F', dtype=np.complex128)
    H = np.zeros((k+1, k), order='F', dtype=np.complex128)
    ritz_vals = []
    # Set q_0 equal to b.
    Q[:,0] = b
    # Normalize q_0.
    Q[:,0] /= sqrt(np.inner(b.conjugate(), b))
    # Perform actual iteration.
    for j in xrange(k):
        # Compute A.dot(q_j).
        Q[:,j+1] = Amul(Q[:,j])
        # Modified Graham Schmidt
        for i in xrange(j+1):
            # Set values of $H$
            H[i,j] = np.inner(Q[:,i].conjugate(), Q[:,j+1])
            Q[:,j+1] -= H[i,j] * Q[:,i]
        # Set subdiagonel element of H.
        H[j+1,j] = sqrt(np.inner(Q[:,j+1].conjugate(), Q[:,j+1]))
        # Stop if ||q_{j+1}|| is too small.
        if abs(H[j+1, j]) < tol:
            # Here I'll copy the arrays to avoid excess memory usage.
            return np.array(H[:j+1,:j], order='F'), np.array(Q[:,j], order='F')
        # Normalize q_{j+1}
        Q[:,j+1] /= H[j+1, j]
    return H, Q

def ritz_compare():
    # A simple exampe of the convergence of the Ritz values
    # to the eigenvalues of the original matrix.
    m = 100
    A = rand(m, m)
    b = rand(m)
    # number of iterations
    k = 40
    # number of eigvals to print
    view_vals = 10
    # run iteration
    H = arnoldi(b, A.dot, k)[0]
    # compute actual eigenvalues
    A_eigs = eig(A, right=False)
    # sort by magnitude
    A_eigs = A_eigs[np.absolute(A_eigs).argsort()[::-1]]
    # print eigvals with largest magnitude
    print A_eigs[:view_vals]
    # compute Ritz Values
    H_eigs = eig(H[:-1], right=False)
    # sort by magnitude
    H_eigs = H_eigs[np.absolute(H_eigs).argsort()[::-1]]
    # print eigvals with largest magnitude
    print H_eigs[:view_vals]
