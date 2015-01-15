import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from scipy.misc import imsave

def PRNG(size, a=1103515245, c=12345, mod=2**31-1, seed=4329):
    x1 = seed
    for x in xrange(43):
        x1 = (x1*a+c) % mod
    random = np.zeros(size)
    random[0] = (x1*a+c) % mod
    for x in xrange(1, size):
        random[x] = (random[x-1]*a+c) % mod
    final = random/float(mod)
    return final


def PRNG1():
    n = 512
    final = PRNG(n**2, 3, 2, 2**16)
    imsave('PRNG1.png', final.reshape(n, n))


if __name__ == "__main__":
    PRNG1()
