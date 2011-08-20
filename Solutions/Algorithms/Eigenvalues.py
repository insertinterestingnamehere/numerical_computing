import scipy as sp
import scipy.sparse as spar
import scipy.linalg as la

def lapacian(A):
    """Compute the lapacian of A and find the second smallest eigenvalue

    INPUTS:
        A   An adjacency array

    RETURNS:
        Q   Lapacian array
        e   second smallest eigenvalue"""
        
    #Calculate the Degree matrix
    D = sp.diag(sp.sum(A, axis=1))

    #if not spar.issparse(A):
    #    A = spar.csr_matrix(A)
    #e = sp.sort(sla.eigs(A, k=2, which="SR", return_eigenvectors=False))[1]
    e = sp.sort(la.eigvals(A))[1]
    
    return (A-D), e

#Random arrays are almost never connected