import bjDump
import bjHelp
import numpy as np
import matplotlib.pyplot as plt
from scipy import rand
import time

def lcg(n,a=1103515245,c=12345,mod=2**31-1,seed=4329):
    X = np.zeros(n)
    X[0] = (a*seed + c)%mod
    for i in range(1,n) :
        X[i] = (a*X[i-1] + c)%mod
    return X/(mod-1)

def between(n,x1,x2,a=1103515245,c=12345,mod=2**31-1,seed=4329):
    return (lcg(n,a,c,mod,seed)*(x2-x1)+x1).astype(int)

def prob4():
    size = 512
    rands = between(size**2,0,255,a=10,c=10,mod=2**30)
    rands = rands.reshape((size,size))
    plt.imshow(rands)
    plt.show()

def prob5():
    N = 10000
    bins = {}
    for i in range(0,N) :
        binNum = toBin(lcg(5,seed=rand(1)).argsort())
        if(bins.has_key(binNum)) : 
            bins[binNum]+= 1
        else:
            bins[binNum] = 1

    spbins = {}
    for i in range(0,N) :
        binNum = toBin(rand(5).argsort())
        if(spbins.has_key(binNum)) : 
            spbins[binNum]+= 1
        else:
            spbins[binNum] = 1
    plt.bar(np.arange(0,120),bins.values())
    plt.show()
    plt.bar(np.arange(0,120),spbins.values(),color='r')
    plt.show()


def findSeedMatch(decks,cards):
   return np.where((decks[:,0:len(cards),0:3]==[ bjHelp.convertToNum(row) for row in cards]).all(axis=1).all(axis=1))[0][0]

def getSweepsEasy(games,n=65536):
    sweeps=np.zeros((n,games,52))
    for i in xrange(n):
        sweeps[i,:]=bjHelp.Suffle(games,2521,13,2**16,i)
    return sweeps


def crackEasy(cards,games):
    sweeps = getSweeps(games)
    hit = findSeedMatch(sweeps,cards)
    return np.array( [bjHelp.convertToName(row) for row in sweeps[hit]] )

def getSweepsHard(games,time,approx=60):
    seeds = np.arange(time-approx,time+approx)
    sweeps=np.array([ bjHelp.Suffle(games,25214903917, 11,2**48,seed) for seed in seeds ])
    return sweeps

def crackHard(cards,games,time=(int)(time.time())):
    sweeps = getSweepsHard(games,time)
    hit = findSeedMatch(sweeps,cards)
    return np.array( [bjHelp.convertToName(row) for row in sweeps[hit]] )
