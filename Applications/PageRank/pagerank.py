import numpy as np
import scipy.sparse as spar
import scipy.linalg as la
from scipy.sparse import linalg as sla

def adj_mat(datafile, n):
    adj = spar.dok_matrix((n,n))
    with open(datafile, 'r') as f:
        for L in f:
            L = L.strip().split()
            try:
                x, y = int(L[0]), int(L[1])
                adj[x, y] = 1
            except:
                continue
    return adj

def dense_pr(data, n=None):
    A = np.asarray(data.tocsr()[:n, :n].todense())
    #data is dense and is of type matrix
    e = np.ones(n)
    for i, v in enumerate(A.sum(1)):
        if v == 0:
            A[i] = e
    
    d = .85
    K = ((1./A.sum(1))[:,np.newaxis]*A).T
    R1 = np.eye(*K.shape)-d*K
    R = la.lstsq(R1, (1-d)*e/float(n))
    max_rank = R[0].max()
    
    return max_rank, np.where(R[0]==max_rank)[0]
  
def sparse_pr(data, n, tol=1e-5):
    A = data.tocsc()[:n, :n]
    s = A.sum(1)
    diag = 1./s
    sinks = s==0
    diag[sinks] = 0
    K = spar.spdiags(diag.squeeze(1), 0, n, n).dot(A).T
    
    d = .85
    convDist = 1
    Rinit = np.ones((n, 1))/float(n)
    Rold = Rinit
    while convDist > tol:
        Rnew = d*K.dot(Rold) + (1-d)*Rinit + (d*Rold[sinks].sum())*Rinit
        convDist = la.norm(Rnew-Rold)
        Rold = Rnew
        
    max_rank = Rnew.max()
    return max_rank, Rnew[Rnew==max_rank]
