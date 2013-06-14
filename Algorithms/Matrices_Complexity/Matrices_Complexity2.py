from scipy import linalg as la
from scipy import sparse as spar
from scipy.sparse import linalg as sla
import scipy as sp

def Problem3(n):
    return la.toeplitz([2,-1]+[0]*(n-2), [2,-1]+[0]*(n-2))

def Problem4(n):
    diags = sp.array([[-1]*n, [2]*n, [-1]*n])
    return spar.spdiags(diags, [-1,0,1], n, n,format='csr')

def Problem5(n, sparse=False):

    b = sp.rand(n,1)
    
    if sparse is True:
        A=Problem4(n)
        return sla.spsolve(A,b)
    else:
        A=Problem3(n)
        return la.solve(A,b)

def Problem6(n):
    A = Problem4(n)
    eig = sla.eigs(A.asfptype(), which="SM")[0].min()
    return eig*(n**2)
    
    
