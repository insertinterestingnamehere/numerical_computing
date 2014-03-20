import scipy as sp
import discretenorm
from matplotlib import pyplot as plt
from mpl_toolkits . mplot3d import Axes3D
from matplotlib import cm
from scipy import stats as st
import tauchenhussey


#Problem 1=====================================================================

def discretenorm(K,mu,sig):
    """
    Discrete approximation of normal density with N bins, mean mu and
    standard deviation sigma.
    """
    pts = np.linspace(mu-3*sig, mu+3*sig, K)
    space = pts[1]-pts[0]
    probs = st.norm.cdf(pts+space/2, loc=mu, scale=sig) - st.norm.cdf(pts-space/2, loc=mu, scale=sig)
    return pts, probs

#Problems 2&3==================================================================
def stochEatCake(beta, N, e_params, W_max=1., plot=False, iid=True):
    #step 1
    if iid:
        e, gamma = discretenorm(*e_params)
    else:
        e, gamma = tauchenhussey(*e_params)
    
    #step 2
    w = np.linspace(0,W_max,N)
    
    #step 3
    v = np.zeros((N,K))
    p = np.zeros((N,K))
    
    #step 4
    actions = np.tile(w, N).reshape((N,N)).T
    actions = actions - actions.T
    actions[actions<0] = 0
    u = np.sqrt(actions)
    u_hat = np.repeat(u,K).reshape((N,N,K))*e
    
    delta = 2.
    while delta >= 1e-9:
        #step 5
        if iid:
            E = (v*gamma).sum(axis=1)
        else:
            E = v.dot(gamma.T)
        
        #step 6
        if iid:
            c = np.swapaxes(np.swapaxes(u_hat, 1, 2)+beta*E, 1, 2)
        else:
            c = u_hat + beta*E
        c[np.triu_indices(N, k=1)] = -1e10
        v_new = np.max(c, axis=1)
        max_indices = np.argmax(c, axis=1)
        p = w[max_indices]
        
        #step 7
        delta = math.sqrt(((v_new - v)**2).sum())
        v = v_new
        
    #step 8
    if plot:
        x = np.arange(0,N)
        y = np.arange(0,K)
        X,Y = np.meshgrid(x,y)
        fig1 = plt.figure()
        ax1 = Axes3D(fig1)
        ax1.plot_surface(w[X], Y, v.T, cmap=cm.coolwarm)
        plt.show()
        
        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        ax2.plot_surface(w[X], Y, p.T, cmap=cm.coolwarm)
        plt.show()
    return v, p

