import scipy as sp
from scipy import stats as st
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import matplotlib.pyplot as plt
from mpl_toolkits . mplot3d import Axes3D
from matplotlib import cm
import discretenorm

def plot_disc_norm():
    x = sp.linspace(-3,3,100);
    y = st.norm.pdf(x,0,1);
    fig, ax = plt.subplots()
    fig.canvas.draw()
    
    ax.plot(x,y)
    
    fill1_x = sp.linspace(-2,-1.5,100)
    fill1_y = st.norm.pdf(fill1_x,0,1)
    fill2_x = sp.linspace(-1.5,-1,100)
    fill2_y = st.norm.pdf(fill2_x,0,1)
    ax.fill_between(fill1_x,0,fill1_y,facecolor = 'blue', edgecolor = 'k',alpha = 0.75)
    ax.fill_between(fill2_x,0,fill2_y,facecolor = 'blue', edgecolor = 'k',alpha = 0.75)
    for label in ax.get_yticklabels():
        label.set_visible(False)
    for tick in ax.get_xticklines():
        tick.set_visible(False)
    for tick in ax.get_yticklines():
        tick.set_visible(False)
    
    plt.rc("font", size = 16)
    plt.xticks([-2,-1.5,-1])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = r"$v_k$"
    labels[1] = r"$\varepsilon_k$"
    labels[2] = r"$v_{k+1}$"
    ax.set_xticklabels(labels)
    plt.ylim([0, .45])

    
    plt.savefig('discnorm.pdf')
    
def plot_stoch_value():    
    #Compute Solution==========================================================
    sigma = .5
    mu = 4*sigma
    K = 7
    Gamma, eps = discretenorm.discretenorm(K,mu,sigma)
    
    N = 100
    W = sp.linspace(0,1,N)
    V = sp.zeros((N,K))
    
    u = lambda c: sp.sqrt(c)
    beta = 0.99
    
    X,Y= sp.meshgrid(W,W)
    Wdiff = Y-X
    index = Wdiff < 0
    Wdiff[index] = 0
    
    util_grid = u(Wdiff)
    
    util3 = sp.tile(util_grid[:,:,sp.newaxis],(1,1,K))
    eps_grid = eps[sp.newaxis,sp.newaxis,:]
    eps_util = eps_grid*util3
    
    Gamma_grid = Gamma[sp.newaxis,:]
    
    delta = 1
    Vprime = V
    z = 0
    while (delta > 10**-9):
        z= z+1
        V = Vprime
        gamV = Gamma_grid*V
        Expval = sp.sum(gamV,1)
        Exp_grid = sp.tile(Expval[sp.newaxis,:,sp.newaxis],(N,1,K))
        arg = eps_util+beta*Exp_grid
        arg[index] = -10^10
        Vprime = sp.amax(arg,1)
        psi_ind = sp.argmax(arg,1)
        psi = W[psi_ind]
        delta = sp.linalg.norm(Vprime - V)
    
    #============================================================    
    #Plot 3D    
    x=sp.arange(0,N)
    y=sp.arange(0,K)
    X,Y=sp.meshgrid(x,y)
    fig1 = plt.figure()
    ax1= Axes3D(fig1)
    ax1.set_xlabel(r'$W$')
    ax1.set_ylabel(r'$\varepsilon$')
    ax1.set_zlabel(r'$V$')
    ax1.plot_surface(W[X],Y,sp.transpose(Vprime), cmap=cm.coolwarm)
    plt.savefig('stoch_value.pdf')
