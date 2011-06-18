import scipy as sp
import row_opers as ops

def LU(A):
    rows, cols = A.shape
    U = A
    L = sp.eye(rows, rows)
    for i in range(rows):
        for j in range(i+1,rows):
            E = ops.cmultadd(rows,j,i,-U[j,i]/U[i,i])
            F = ops.cmultadd(rows,j,i,U[j,i]/U[i,i])
            U = sp.dot(E,U)
            L = sp.dot(L,F)
    return (L,U)
