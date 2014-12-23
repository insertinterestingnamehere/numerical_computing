import numpy as np
import matplotlib.pylab as plt
from scipy import linalg as la

def PCA(dat, center=False, percentage=0.8):
    M = dat[:,0].size
    N = dat[0,:].size
    if center:
        mu = np.mean(dat,0)
        dat -= mu

    U, L, Vh = la.svd(dat, full_matrices=False)
    
    V = (Vh.T).conjugate()
    SIGMA = np.diag(L)
    X = U.dot(SIGMA)
    Lam = L**2

    sLam = Lam.sum()
    csum = [Lam[:i+1].sum()/sLam for i in xrange(N)]

    normalized_eigenvalues = Lam/sLam
    for i, x in np.ndenumerate(csum):
        if not x < percentage :
            n_components = i[0]
            break
        
    return normalized_eigenvalues, 
            V[:,0:n_components],
            SIGMA[0:n_components,0:n_components],
            X[:,0:n_components]

def scree(normalized_eigenvalues):
    plt.plot(normalized_eigenvalues, 'b-', normalized_eigenvalues, 'bo')
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of Variance")
    plt.show()
