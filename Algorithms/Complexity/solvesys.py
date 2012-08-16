import scipy as sp
from scipy import linalg as la

def solveSys(A, b):
    return la.solve(A,b)
    
def invSys(A, b):
    return sp.dot(la.inv(A), b)

def smwSys(A, b, n, u, v):
    Ainv = sp.eye(n) - sp.dot(u,v)/(1+sp.dot(v,u))
    return sp.dot(Ainv, b)
