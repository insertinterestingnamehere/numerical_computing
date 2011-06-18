import scipy as sp

def arrayMult(A,B):
    a, b = A.shape
    c, d = B.shape
    C = sp.zeros((a,d))
    for i in xrange(a):
        for j in xrange(b):
            for k in xrange(d):
                C[i,k]+=A[i,j]*B[j,k]
    return C
    
def arrayMult2(A, B):
    a, b = A.shape
    c, d = B.shape
    C = sp.zeros((a,d))
    for i in xrange(a):
        for k in xrange(d):
            C[i,k] = sp.dot(A[i,:], B[:,k])
    return C