import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file("../../matplotlibrc")

import matplotlib.pyplot as plt
from scipy.linalg import svd

def svals(img):
    U, s, Vt = svd(img)
    plt.plot(s)
    plt.savefig('hubble_svals.pdf')

if __name__ == "__main__":
    img = plt.imread('hubble_red.png')
    svals(img)