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

#naive way of suffling the deck
def shuffle(deck):
    size = len(deck)
    newdeck=np.empty_like(deck)
    for i in xrange(size/2):
         if np.random.randint(0, 1) == 0:
            newdeck[i*2]=deck[i]
            newdeck[i*2+1]=deck[size/2+i]
         else:
            newdeck[i*2]=deck[size/2+i]
            newdeck[i*2+1]=deck[i]
    deck[:] = newdeck

#Better way
def shuffleB(deck):
    s = len(deck)/2
    ra = np.random.randint(0, 2, s)
    #Genrates random number half the size of the deck. 
    #if ra[i]==0 then the ith card form the second half of the deck goes in the first slot in the ith place in the new deck
    #and the ith card in the first half goes in the the secod slot in the ith place in the new deck 
    tempdeck = deck[range(s) + (1-ra)*s].copy()
    deck[::2] = deck[range(s) + ra*s]
    deck[1::2] = tempdeck
    return deck

# image vectorization problem
def image_vect():
	# a
	I = rand(100,200,3)
	I.mean(axis=2)
	# b
	return np.absolute(I-I.mean(axis=2, keepdims=True))

#edit==1 inverts
#edit==2 greyscales
#edit==3 does a motion blur of n
def imageEditor(X,edit,n=1):
    if edit==1:
        Xnew = 255 - X
    elif edit == 2:
        Xnew = X.copy()
        #The [:, :, np.newaxis] turns it the 2-d array into a 3-d array
        Xnew[:, :, :] = (X/3).sum(2)[:, :, np.newaxis]
    else:
        Xnew = X.copy()/n
        for i in xrange(1, n):
            Xnew[:,:-i,:] += X[:,i:,:] / n
            Xnew[:,-i:,:] += Xnew[:,-i:,:] / n
            
    plt.imshow(Xnew)
    plt.show()
