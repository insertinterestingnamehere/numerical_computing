from scipy import linalg as la
from scipy import sparse as spar
import scipy as sp

def Problem1(n):
    """Use linalg.toeplitz() and linalg.triu() to generate matrices of arbitrary size"""

    from scipy.linalg import triu


    ut = triu([[0]*i+[x for x in range(1,(n+1)-i)] for i in range(n)])
    toep = la.toeplitz([1.0/(i+1) for i in range(n)])

    return ut, toep

def Problem4(n):
    return la.toeplitz([2,-1]+[0]*(n-2), [2,-1]+[0]*(n-2))

def Problem5(n):
    diags = sp.array([[-1]*n, [2]*n, [-1]*n])
    return spar.spdiags(diags, [-1,0,1], n, n,format='csr')

def Problem6(n, sparse=False):

    b = sp.rand(n,1)
    
    if sparse is True:
        from scipy.sparse import linalg as sla
        A=Problem5(n)
        return sla.spsolve(A,b)
    else:
        A=Problem4(n)
        return la.solve(A,b)

def Problem7(n):
    A = Problem4(n)
    eig = la.eigvals(A).min()
    return eig*(n**2)
    
    