import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt

def wilkinson_poly():
    """ Reproduce the Wilkinson Polynomial example shown in the lab. """
    roots = np.arange(1,21)
    w_coeffs = np.array([1, -210, 20615, -1256850, 53327946, -1672280820,
                    40171771630, -756111184500, 11310276995381,
                    -135585182899530, 1307535010540395,
                    -10142299865511450, 63030812099294896,
                    -311333643161390640, 1206647803780373360,
                    -3599979517947607200, 8037811822645051776,
                    -12870931245150988800, 13803759753640704000,
                    -8752948036761600000, 2432902008176640000])
                    
    perturb = np.zeros(21)
    perturb[1]=1e-7
    rootsp = np.roots(np.poly1d(w_coeffs+perturb))

    # Plot original roots
    plt.scatter(range(1,21), np.zeros(20), s=30)
    
    # Plot roots of the perturbed polynomial
    plt.scatter(np.real(rootsp), np.imag(rootsp), s=60, c='red', marker='x')
    plt.savefig('wilkinsonpolynomial.pdf', bbox_inches='tight')
    plt.close()

def plot_eig_condit():
    x0, x1 =-100, 100
    y0, y1 =-100, 100
    res=200
    x = np.linspace(x0,x1,res)
    y = np.linspace(y0,y1,res)
    X,Y = np.meshgrid(x,y)
    J = np.empty_like(X)
    for i in xrange(res):
        for j in xrange(res):
            M = np.array([[1, X[i,j]],[Y[i,j],1]])
            eigs = la.eig(M)[0]
            perturb = np.random.normal(0, 1e-6, M.shape) + np.random.normal(0,1e-6, M.shape)*1j
            eigsp = la.eig(M+perturb)[0]
            k = la.norm(eigs-eigsp)/la.norm(perturb)
            J[i,j] = k*la.norm(M)/la.norm(eigs)
    plt.pcolormesh(X,Y,J, cmap='Greys')
    plt.colorbar()
    plt.savefig('eigenvalue_conditioning.png', bbox_inches='tight')
    plt.close()
    
def wilkinson_many():
    roots = np.arange(1,21)
    w_coeffs = np.array([1, -210, 20615, -1256850, 53327946, -1672280820,
                    40171771630, -756111184500, 11310276995381,
                    -135585182899530, 1307535010540395,
                    -10142299865511450, 63030812099294896,
                    -311333643161390640, 1206647803780373360,
                    -3599979517947607200, 8037811822645051776,
                    -12870931245150988800, 13803759753640704000,
                    -8752948036761600000, 2432902008176640000])
    for trial in xrange(100):
        perturb = np.random.normal(1, 1e-10, 21)
        rootsp = np.roots(np.poly1d(w_coeffs*perturb))

        # Plot roots of the perturbed polynomial
        plt.scatter(np.real(rootsp), np.imag(rootsp), c='black', s=5, marker='.')

    # Plot original roots
    plt.scatter(range(1,21), np.zeros(20), s=30)
    plt.xlim(0, 23)
    plt.savefig('wilkinsonpolynomial_many.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    wilkinson_poly()
    wilkinson_many()
    plot_eig_condit()