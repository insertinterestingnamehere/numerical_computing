import numpy as np
import matplotlib.pyplot as plt

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
