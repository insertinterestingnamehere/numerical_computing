import bjCommon
import numpy as np
import matplotlib.pyplot as plt
from scipy import rand
import time


#Problem 1
def lcg(n,a=1103515245,c=12345,mod=2**31-1,seed=4329):
    """
    Return an array of random numbers using the parameter specified
    """
    X = np.zeros(n)
    X[0] = (a*seed + c)%mod
    for i in range(1,n) :
        X[i] = (a*X[i-1] + c)%mod
    return X/(mod-1)

#Problem 2
def between(n,x1,x2,a=1103515245,c=12345,mod=2**31-1,seed=4329):
    """
    Scale n to be between x1 and x2
    """
    return (lcg(n,a,c,mod,seed)*(x2-x1)+x1).astype(int)

def problem3():
    """
    Graph the randoms from our lcg and also from scipy.rand
    """
    
    plt.subplot(211)
    size = 512
    rands = between(size**2,0,255,a=11035,c=1004,mod=2**15)
    rands = rands.reshape((size,size))
    plt.imshow(rands)
    plt.show()
	plt.close()
    
    plt.subplot(212)
    plt.imshow(rand(512,512)*255)
    plt.show()
	plt.close()

#Problem 4
def getSweepsEasy(games,n):
    """
    Return an n x games x 52 Array that represents the shuffles for <games> for
    the first n shuffles
    """
    #mine returns an np.array just we can use array slicing
    return np.array([bjCommon.shuffle(games,2521,13,2**16,seed) for seed in xrange(n)])

#Problem 5
def crackBlackJack(sweeps,cardTuples):
    """
    Return the sweeps whose first three cards from the nth game match the nth
    3-tuple in cardTuples
    
    For example, if cardTuples is a 2 x 3 list of tuples crackBlackJack should 
    return the sweeps in "sweeps" whose first 3 cards in the first game match
    cardTyples[0] and whose first 3 cards in the second game match cardTuples[1]
    """
    for game in range(len(cardTuples)):
        hits = bjCommon.findSeedMatch(sweeps,game,cardTuples[game])
        sweeps = sweeps[hits]
    return bjCommon.convertToName(sweeps)

#Problem 6
def getSweepsHard(games,time,approx=120):
    """
    Return an n x games x 52 Array that represents the shuffles for <games> for
    the n possible seeds from the 2*approx second interval (above and below time)
    """
    #Create a list of seeds based on the time interval
    seeds = np.arange(time-approx,time+approx)
    return np.array([bjCommon.shuffle(games,25214903917, 11,2**48,seed) for seed in seeds])

