import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse

def spy_sparse():
    n = 10000
    B = np.random.rand(3, n)
    A = sparse.spdiags(B, range(-1, 2), n, n)
    plt.spy(A)
    plt.savefig('spy.pdf')
    
spy_sparse()
