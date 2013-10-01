import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

# Assorted vectorization problems
def assortment():
    # a
    X = rand(100, 10)
    X.dot(X.T)
    # b
    (X*X).sum(axis=1)
    # c
    A = rand(10, 10)
    V = rand(100, 10)
    (V.dot(A)*V).sum(axis=1)
    # d
    A = rand(1000)
    (A<.5).sum()
    # e
    A[A<.25] = 0.
    # f
    A = rand(10, 10)
    X = rand(100, 10)
    A.dot(X.T).T
    # g
    A = rand(10, 2, 2)
    B = rand(20, 2)
    A.dot(B.T).swapaxes(1, 2)
    # h
    A = rand(100, 100)
    (A[:,0] < .5).dot(A).sum()
    # i
    P = rand(100)
    D = (rand(100, 100) < .5)
    P * D.sum(axis=1) - D.dot(P)
    # j (shuffle problem)
    A = np.arange(52)
    A[::2], A[1::2] = A[A.shape[0]//2:].copy(), A[:A.shape[0]//2].copy()

# image vectorization problem
def image_vect():
	# a
	I = rand(100, 200, 3)
	I.mean(axis=2)
	# b
	return np.absolute(I - I.mean(axis=2, keepdims=True))

#edit==1 inverts
#edit==2 grayscales
#edit==3 does a motion blur of n
def imageEditor(X,edit,n=1):
    if edit==1:
        Xnew = 255 - X
    elif edit == 2:
        Xnew = X.mean(axis=2, dtype=int)
    else:
        Xnew = X.copy()/n
        for i in xrange(1, n):
            Xnew[:,:-i,:] += X[:,i:,:] / n
            Xnew[:,-i:,:] += Xnew[:,-i:,:] / n
            
    plt.imshow(Xnew)
    plt.show()
