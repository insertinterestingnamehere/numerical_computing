import numpy as np
from scipy import linalg as la
#import matplotlib
#matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

def plot_gmres():
    M = 2*np.eye(200)+np.random.normal(0, .5/np.sqrt(200), (200,200))
    b = np.ones(200)
    k = 200
    tol = 1e-8
    res = np.empty(k)
    
    Q = np.empty((b.size, k+1))
    H = np.zeros((k+1, k))
    bnorm = la.norm(b)
    Q[:,0] = b/bnorm
    for j in xrange(k):
        # Arnoldi algorithm, minus the last step
        Q[:,j+1] = M.dot(Q[:,j])
        for i in xrange(j+1):
            H[i,j] = np.inner(Q[:,i].conjugate(), Q[:,j+1])
            Q[:,j+1] -= H[i,j] * Q[:,i]
        H[j+1,j] = la.norm(Q[:,j+1])
        
        # Calculate least squares solution
        y, res[j] = la.lstsq(H[:j+2, :j+1], bnorm*np.eye(j+2)[0] )[:2]
        #calculate residual
        res[j] = np.sqrt(res[j])/bnorm
        
        # Stop if the residual is small enough OR if Arnoldi has terminated.
        # Though, I think the second condition implies the first.
        if res[j] < tol or H[j+1, j] < tol:
            break
        
        # Last step of Arnoldi
        Q[:,j+1] /= H[j+1, j]
    plt.subplot(1,2,2)
    plt.plot(xrange(j+1), res[:j+1] )
    plt.gca().set_yscale('log')
    
    # plot eigenvalues
    evals, evecs = la.eig(M)
    plt.subplot(1,2,1)
    plt.scatter(np.real(evals), np.imag(evals))
    plt.savefig('./plot_gmres.pdf', bbox_inches='tight')
    plt.close()