import scipy as sp

def hilbert(n):
    """Calculate an nxn Hilbert matrix"""

    H = sp.zeros((n,n))

    for i in range(n):
        for j in range(n):
            H[i][j] = 1.0/(i+j+1)

    return H

def cond(matrix):
    """Calculate the condition number of matrix"""

    return sp.linalg.norm(matrix)*sp.linalg.norm(sp.linalg.pinv(matrix))

