import scipy as sp
import scipy.linalg as la


def Problem2(Q,b,x0,tol=1e-10):
    """Use conjugate gradient method to minimize Ax-b"""
    
    x = x0.copy()
    g = Q.dot(x)-b
    d = -g
    
    ndim = x.size
    
    for iter in range(ndim):
        if la.norm(g)<tol:
            break
        a = -1.0*(g.T.dot(d))/(d.T.dot(Q).dot(d))
        x += a*d
        g = Q.dot(x)-b
        beta = (g.T.dot(Q).dot(d))/(d.T.dot(Q).dot(d))
        d = -g+beta*d
        
    return x

def Problem1(A,b,x0,tol=1e-7,maxiter=5000):
    """Find the minimum of a function"""
    
    x = x0.copy()
    Q = A.T.dot(A)
    niter = maxiter
    while la.norm(A.dot(x)-b)>tol and maxiter>0:
        grad_f = 2.0*A.T.dot(A.dot(x)-b)
        alpha_f = (grad_f.T.dot(grad_f))/(4.0*grad_f.T.dot(Q.dot(grad_f)))
        x = x + -1.0*alpha_f*grad_f
        niter -= 1
        
    return x, maxiter-niter
    
"""    
Explanations:
    We found that 4.0*denominator of alpha_f converged faster than just having the denominator.
    
    I wasn't able to get the eigenvector of A.T.dot(A) to converge in 1 iteration.  The lowest convergence happened in 236 iterations
    
"""