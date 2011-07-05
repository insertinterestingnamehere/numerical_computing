import scipy as sp
import matplotlib.pyplot as plt


def readimg(filename, channel=1):
    return sp.flipud(filename)[:,:,channel]

def compressSVD(U, s, Vt, rank):
    return sp.dot(sp.dot(U[:,0:n], s[0:n, 0:n]), Vt[0:n,:])

def Sdiag(x, shape):
    S = sp.zeros(shape)
    S[0:x.size, 0:x.size] = sp.diag(x)
    return S