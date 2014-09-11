import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt

def wilkinson_poly():
    """ Reproduce the Wilkinson Polynomial example shown in the lab. """
    roots = np.arange(1, 21)
    cfs = np.array([1, -210, 20615, -1256850, 53327946, -1672280820,
                    40171771630, -756111184500, 11310276995381,
                    -135585182899530, 1307535010540395,
                    -10142299865511450, 63030812099294896,
                    -311333643161390640, 1206647803780373360,
                    -3599979517947607200, 8037811822645051776,
                    -12870931245150988800, 13803759753640704000,
                    -8752948036761600000, 2432902008176640000])
    peturbation = np.random.normal(scale=1E-5, size=cfs.size)
    computed_roots = np.poly1d(cfs + peturbation).roots
    plt.scatter(roots.real, roots.imag, marker='D')
    plt.scatter(computed_roots.real, computed_roots.imag, c='red', marker='x')
    plt.savefig("wilkinsonpolynomial.pdf")
    plt.clf()

if __name__ == "__main__":
    wilkinson_poly()