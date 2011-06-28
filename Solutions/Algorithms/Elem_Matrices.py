from scipy import eye, dot

def rowswap(n, j, k):
    """Swaps two rows
        INPUTS: n -> matrix size
        j, k -> the two rows to swap"""
    out = eye(n)
    out[j,j]=0
    out[k,k]=0
    out[j,k]=1
    out[k,j]=1
    return out
    
def cmult(n, j, const):
    """Multiplies a row by a constant
    INPUTS: n -> array size
            j -> row
            const -> constant"""
    out = eye(n)
    out[j,j]=const
    return out
    
def cmultadd(n, j, k, const):
    """Multiplies a row (k) by a constant and adds the result to another row (j)"""
    out = eye(n)
    out[j,k] = const
    return out
    
def ref(A):
    """Performs a naive row reduction on A"""
    
    n = min(A.shape)
    ref = A.astype(float).copy()
    for a in xrange(n):
        for b in xrange(a,n):
            if ref[a,a] != 0:
                ref = dot(cmultadd(n, b, a, -ref[b, a]/ref[a, a]), ref)
            else: continue
        ref = dot(cmult(n, a, 1.0/ref[a,a]), ref)
        
    return ref

def LU(A):
    rows, cols = A.shape
    U = A
    L = eye(rows, rows)
    for i in range(rows):
        for j in range(i+1,rows):
            E = cmultadd(rows,j,i,-U[j,i]/U[i,i])
            F = cmultadd(rows,j,i,U[j,i]/U[i,i])
            U = dot(E,U)
            L = dot(L,F)
    return (L,U)

def LU_det(A):
    """Find the determinant of a matrix using the LU factorization"""
    U = LU(A)[1]
    det = 1
    for i in range(U.shape[0]):
        det *= U[i,i]
    return det