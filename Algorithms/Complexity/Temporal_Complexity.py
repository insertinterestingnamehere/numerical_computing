from timer import timer
import matplotlib.pyplot as plt
from scipy import linalg as la
import scipy as sp

def MatrixAdd(A,B):
    return A+B

def MatrixInv(A):
    return la.inv(A)

def MatrixDet(A):
    return la.det(A)

def MatrixLU(A):
    return la.lu(A)

def MatrixSVD(A):
    return la.svd(A)

def MatrixSolve(A,b):
    return la.solve(A,b)

def main(f):
    i = sp.arange(1500, 2500+200, 200)

    y = []
    with timer() as t:
        for n in i:
            if f == MatrixAdd:
                t.time(f, sp.rand(n,n), sp.rand(n,n))
            elif f == MatrixSolve:
                t.time(f, sp.rand(n,n), sp.rand(n,1))
            else:
                t.time(f, sp.rand(n,n))
            y.append(t.results[0][0])

    X = sp.row_stack([sp.log(i), sp.ones_like(i)])
    sol = la.lstsq(X.T, sp.log(y))
    print sol[0][0]
    plt.loglog(i,y)
    plt.show()
