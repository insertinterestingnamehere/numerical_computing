import numpy as np
from numpy.random import rand
from scipy.linalg import eig
from cmath import sqrt
from pyfftw.interfaces.scipy_fftpack import fft
from scipy import sparse as ss
from scipy.sparse.linalg import eigsh

# arnoldi iteration
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
            # Here I'll copy the array to avoid excess memory usage.
            return np.array(H[:j,:j], order='F')
        # Normalize q_{j+1}
        Q[:,j+1] /= H[j+1, j]
    return H[:-1]

# Ritz Value Convergence
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
    H = arnoldi(b, A.dot, k)
    # compute actual eigenvalues
    A_eigs = eig(A, right=False)
    # sort by magnitude
    A_eigs = A_eigs[np.absolute(A_eigs).argsort()[::-1]]
    # print eigvals with largest magnitude
    print A_eigs[:view_vals]
    # compute Ritz Values
    H_eigs = eig(H, right=False)
    # sort by magnitude
    H_eigs = H_eigs[np.absolute(H_eigs).argsort()[::-1]]
    # print eigvals with largest magnitude
    print H_eigs[:view_vals]

# Fourier Transform Eigenvalues
def fft_eigs():
    # Returns an estimate for the eigenvalues of the
    # Discrete Fourier Transform.
    m = 2**20
    b = rand(m)
    k = 10
    H = arnoldi(b, fft, k)
    H_eigs = eig(H, right=False)
    H_eigs /= sqrt(m)
    H_eigs = H_eigs[np.absolute(H_eigs).argsort()][::-1]
    return H_eigs[:10]

# needed for the polynomial root finding problem
def companion_multiply(c, u):
    v = np.empty_like(u)
    v[0] = - c[0] * u[-1]
    v[1:] = u[:-1] - c[1:] * u[-1]
    return v

# Polynomial root finding
def root_find():
    m = 1000
    k = 50
    disp = 5
    c = rand(m)
    p = np.poly1d([1] + list(c[::-1]))
    Cmul = lambda u: companion_multiply(c, u)
    b = rand(m)
    H = arnoldi(b, Cmul, k)
    H_eigs = eig(H, right=False)
    H_eigs = H_eigs[np.absolute(H_eigs).argsort()][::-1]
    print H_eigs[:disp]
    roots = p.roots
    roots = roots[np.absolute(roots).argsort()][::-1]
    print roots[:disp]

# Lanczos Iteration
def lanczos(b, Amul, k, tol=1E-8):
    """Perform basic Lanczos Iteration given a starting vector 'b',
    a function 'Amul' representing multiplication by some matrix A,
    a number 'k' of iterations to perform, and a tolerance 'tol' to
    determine if the algorithm should stop early."""
    # Some Initialization
    # We will use $q_0$ and $q_1$ to store the needed $q_i$
    q0 = 0
    q1 = b / sqrt(np.inner(b.conjugate(), b))
    alpha = np.empty(k)
    beta = np.empty(k)
    beta[-1] = 0.
    # Perform the iteration.
    for i in xrange(k):
        # z is a temporary vector to store q_{i+1}
        z = Amul(q1)
        alpha[i] = np.inner(q1.conjugate(), z)
        z -= alpha[i] * q1
        z -= beta[i-1] * q0
        beta[i] = sqrt(np.inner(z.conjugate(), z)).real
        # Stop if ||q_{j+1}|| is too small.
        if beta[i] < tol:
            return alpha[:i+1].copy(), beta[:i].copy()
        z /= beta[i]
        # Store new q_{i+1} and q_i on top of q0 and q1
        q0, q1 = q1, z
    return alpha, beta[:-1]

# Needed for tridiagonal eigenvalue problem
def tri_mul(alpha, beta, u):
    v = alpha * u
    v[:-1] += beta * u[1:]
    v[1:] += beta * u[:-1]
    return v

# tridiagonal eigenvalue problem
def tridiag_eigs():
    # Most of this code is just constructing
    # tridiagonal matrices and calling functions
    # they have already written.
    m = 1000
    k = 100
    A = np.zeros((m, m))
    a = rand(m)
    b = rand(m-1)
    np.fill_diagonal(A, a)
    np.fill_diagonal(A[1:], b)
    np.fill_diagonal(A[:,1:], b)
    Amul = lambda u: tri_mul(a, b, u)
    alpha, beta = lanczos(rand(m), Amul, k)
    H = np.zeros((alpha.size, alpha.size))
    np.fill_diagonal(H, alpha)
    np.fill_diagonal(H[1:], beta)
    np.fill_diagonal(H[:,1:], beta)
    H_eigs = eig(H, right=False)
    H_eigs.sort()
    H_eigs = H_eigs[::-1]
    print H_eigs[:10]
    A = np.zeros((m, m))
    np.fill_diagonal(A, a)
    np.fill_diagonal(A[1:], b)
    np.fill_diagonal(A[:,1:], b)
    A_eigs = eig(A, right=False)
    A_eigs.sort()
    A_eigs = A_eigs[::-1]
    print A_eigs[:10]

# algebraic connectivity problem
def verify_connected():
    # The idea here is to notice that only
    # one eigenvalue is nearly equal to 0.
    m = 1000
    d = np.ones(m)
    d[1:-1] += np.ones(m-2)
    l = ss.diags([-np.ones(m-1), d, -np.ones(m-1)], [-1, 0, 1])
    return eigsh(l, which='SM', return_eigenvectors=False)

def verify_disconnected():
    # The idea here is to notice that two of
    # the eigenvalues are nearly equal to 0.
    m = 1000
    cut = 500
    d = np.ones(m)
    d[1:-1] += np.ones(m-2)
    d1 = -np.ones(m-1)
    d1[cut] = 0
    d[[cut, cut+1]] =1
    l = ss.diags([d1, d, d1], [-1, 0, 1])
    return eigsh(l, which='SM', return_eigenvectors=False)
