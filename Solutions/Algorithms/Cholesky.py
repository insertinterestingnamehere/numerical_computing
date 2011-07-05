import scipy as sp


#def PseudoCholesky(Matrix):
#   nrows, ncols = Matrix.shape
#   Lmat = zeros_like(Matrix)
#   for i in nrows:
#        for j in ncols:
#            if i > j:
#                Lmat[i,j] = (1.0/L[j,j])*(Matrix[i,j]-sum((Lmat[i,k]*Lmat[j,k].conjugate() for k in range(j)))
#            elif i==j:
#                Lmat[i,i] = sqrt(Matrix[i,i]-sum((Lmat[i,k]*Lmat[i,k].conjugate() for k in range(i))))
#
#    return Lmat, Lmat.T.conjugate()

def cholesky(A):
    """Calculate the Cholesky decomposition of a positive-definite matrix"""
    nrows, ncols = A.shape
    Lmat = sp.zeros_like(A)
    for i in range(nrows):
        for j in range(ncols):
            if i > j:
                Lmat[i,j] = (1.0/Lmat[j,j])*(A[i,j]-sum((Lmat[i,k]*Lmat[j,k].conjugate() for k in range(j))))
            elif i==j:
                Lmat[i,i] = sp.sqrt(A[i,i]-sum((Lmat[i,k]*Lmat[i,k].conjugate() for k in range(i))))

    return Lmat, Lmat.T.conjugate()

