import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file("../../matplotlibrc")

import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np

def svals(img):
    U, s, Vt = svd(img)
    plt.plot(s)
    plt.savefig('hubble_svals.pdf')

def lowrank(img, rankvals):
	U, s, Vt = svd(img)
	
	for n in rankvals:
		u1, s1, vt1 = U[:,0:n], np.diag(s[0:n]), Vt[0:n,:]
		plt.imsave("rank{}.png".format(n), u1.dot(s1).dot(vt1))


if __name__ == "__main__":
    img = plt.imread('hubble_red.png')
    svals(img)
    lowrank(img, [1, 14, 27, 40])
