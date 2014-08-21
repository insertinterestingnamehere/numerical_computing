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

def PRNGint(size, least=0, greatest=2, a=1103515245, c=12345, mod=2**31-1, seed=432946458):
    final = PRNG(size, a, c, mod, seed)
    final = (final*(greatest-least)).astype(int) + least
    return final

def PRNG1():
    n = 512
    m = 2
    final = PRNGint(n**2, 0, m)
    imsave('PRNG1.png', final.reshape(n, n))

def PRNG2():
    n = 2**9
    final = PRNG(n**2, 25214903917, 11, 2**48, 2*17+7)
    imsave('PRNG2.png', final.reshape(n, n))


if __name__ == "__main__":
    PRNG1()
    PRNG2()