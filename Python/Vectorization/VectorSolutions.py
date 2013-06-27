import numpy as np
import matplotlib.pyplot as plt

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

def shuffleB(deck):
    s = len(deck)/2
    ra = np.random.randint(0, 2, s)
    tempdeck = deck[range(s) + (1-ra)*s].copy()
    deck[::2] = deck[range(s) + ra*s]
    deck[1::2] = tempdeck
    return deck

def imageEditor(X,edit,n=1):
    if edit==1:
        Xnew = 255 - X
    elif edit == 2:
        Xnew = X.copy()
        Xnew[:, :, :] = (X/3).sum(2)[:, :, np.newaxis]
    else:
        Xnew = X.copy()/n
        for i in xrange(1, n):
            Xnew[:,:-i,:] += X[:,i:,:] / n
            Xnew[:,-i:,:] += Xnew[:,-i:,:] / n
            
    plt.imshow(Xnew)
    plt.show()

def where_thres(X, thres=128):
    Xnew = np.where(X<thres, 0, 255)
    plt.imshow(Xnew)
    plt.show()
