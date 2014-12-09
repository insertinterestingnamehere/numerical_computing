import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

def PCA(dat, center=False, percentage=0.8):
    M, N = dat.shape
    if center:
        mu = np.mean(dat,0)
        dat -= mu

    U, L, Vh = la.svd(dat, full_matrices=False)
    
    V = Vh.T.conjugate()
    SIGMA = np.diag(L)
    X = U.dot(SIGMA)
    Lam = L**2

    normalized_eigenvalues = Lam/Lam.sum(dtype=float)
    csum = [normalized_eigenvalues[:i+1].sum() for i in xrange(N)]
    n_components = [x < percentage for x in csum].index(False) + 1

    return (normalized_eigenvalues, 
            V[:,0:n_components], 
            SIGMA[0:n_components,0:n_components], 
            X[:,0:n_components])

def scree(normalized_eigenvalues):
    fig = plt.figure()
    plt.plot(normalized_eigenvalues,'b-', normalized_eigenvalues, 'bo')
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of Variance")
    return fig
    
