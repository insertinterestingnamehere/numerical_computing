from scipy import *
#from pylab import *
import matplotlib.pylab as plt
from scipy import linalg as la

def PCA(dat,center=False,percentage=0.8):
    M=dat[:,0].size
    N=dat[0,:].size
    if center:
	    mu = mean(dat,0)
	    dat -= mu

    U,L,Vh = la.svd(dat,full_matrices=False)
    
    V = (Vh.T).conjugate()
    SIGMA = diag(L)
    X = dot(U,SIGMA)
    Lam = L**2

    csum = [sum(Lam[:i+1])/sum(Lam) for i in range(N)]

    normalized_eigenvalues = Lam/sum(Lam)
    n_components = array([x < percentage for x in csum]).tolist().index(False)

    return normalized_eigenvalues,V[:,0:n_components],SIGMA[0:n_components,0:n_components],X[:,0:n_components]

def scree(normalized_eigenvalues):
    plt.plot(normalized_eigenvalues,'b-',normalized_eigenvalues,'bo')
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of Variance")
    plt.show()
    return
