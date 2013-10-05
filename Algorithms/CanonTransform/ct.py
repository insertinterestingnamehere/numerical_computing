import numpy as np
from scipy import linalg as la
from math import copysign

def hqr(A):
    """Finds the QR decomposition of A using Householder reflectors.
    input: 	A, mxn array with m>=n
    output: Q, orthogonal mxm array
            R, upper triangular mxn array
            s.t QR = A
    """
    # This is just a pure Python implementation.
    # It's not fully optimized, but it should
    # have the right asymptotic speed.
    # initialize Q and R
    # start Q as an identity
    # start R as a C-contiguous copy of A
    # take a transpose of Q to start out
    # so it is C-contiguous when we return the answer
    Q = np.eye(A.shape[0]).T
    R = np.array(A, order="C")
    # initialize m and n for convenience
    m, n = R.shape
    # avoid reallocating v in the for loop
    v = np.empty(A.shape[1])
    for k in xrange(n-1):
        # get a slice of the temporary array
        vk = v[k:]
        # fill it with corresponding values from R
        vk[:] = R[k:,k]
        # add in the term that makes the reflection work
        vk[0] += copysign(la.norm(vk), vk[0])
        # normalize it so it's an orthogonal transform
        vk /= la.norm(vk)
        # apply projection to R
        R[k:,k:] -= 2 * np.outer(vk, vk.dot(R[k:,k:]))
        # Apply it to Q
        Q[k:] -= 2 * np.outer(vk, vk.dot(Q[k:]))
    # note that its returning Q.T, not Q itself
    return Q.T, R

def hess(A):
    """Computes the upper Hessenberg form of A using Householder reflectors.
    input:  A, mxn array
    output: Q, orthogonal mxm array
            H, upper Hessenberg
            s.t. QHQ' = A
    """
    # similar approach as the householder function.
    # again, not perfectly optimized, but good enough.
    Q = np.eye(A.shape[0]).T
    H = np.array(A, order="C")
    # initialize m and n for convenience
    m, n = H.shape
    # avoid reallocating v in the for loop
    v = np.empty(A.shape[1]-1)
    for k in xrange(n-2):
        # get a slice of the temporary array
        vk = v[k:]
        # fill it with corresponding values from R
        vk[:] = H[k+1:,k]
        # add in the term that makes the reflection work
        vk[0] += copysign(la.norm(vk), vk[0])
        # normalize it so it's an orthogonal transform
        vk /= la.norm(vk)
        # apply projection to H on the left
        H[k+1:,k:] -= 2 * np.outer(vk, vk.dot(H[k+1:,k:]))
        # apply projection to H on the right
        H[:,k+1:] -= 2 * np.outer(H[:,k+1:].dot(vk), vk)
        # Apply it to Q
        Q[k+1:] -= 2 * np.outer(vk, vk.dot(Q[k+1:]))
    # note that its returning Q.T, not Q itself
    return Q.T, H
