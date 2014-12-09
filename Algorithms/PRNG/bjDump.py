import numpy as np
import time

def getDeck():
    hearts=['Aheart','2heart','3heart','4heart','5heart','6heart','7heart','8heart','9heart','10heart','Jheart','Qheart','Kheart']
    spades=['Aspade','2spade','3spade','4spade','5spade','6spade','7spade','8spade','9spade','10spade','Jspade','Qspade','Kspade']
    clubs=['Aclub','2club','3club','4club','5club','6club','7club','8club','9club','10club','Jclub','Qclub','Kclub']
    diamonds=['Adiamond','2diamond','3diamond','4diamond','5diamond','6diamond','7diamond','8diamond','9diamond','10diamond','Jdiamond','Qdiamond','Kdiamond']
    Deck=hearts+diamonds+clubs+spades
    return Deck

def dumpEasy(numberOfRounds=1):
    DeckArray = np.array(getDeck())
    PlayDecks = []
    seed=int(time.time())
    for x in range(numberOfRounds):
        PlayDecks.append(Suffle(DeckArray,(x+1)*((seed%50+(seed+1)%2)),a=2521,c=13,mod=2**16,seed=seed).tolist())
    return np.array(PlayDecks)

def dumpHard(numberOfRounds=1):
    DeckArray = np.array(getDeck())
    PlayDecks = []
    fakeSeed=int(time.time())
    seed = fakeSeed + fakeSeed%120 - 60
    for x in range(numberOfRounds):
        PlayDecks.append(Suffle(DeckArray,x+1,a=25214903917,c=11,mod=2**48,seed=seed).tolist())
    return fakeSeed,np.array(PlayDecks)
            
def PRNG(size,a=25214903917,c=11,mod=2**48,seed=2*17+7):
    x1=seed
    for x in range(43):
        x1=(x1*a+c)%mod
    random=np.zeros(size)
    random[0]=(x1*a+c)%mod
    for x in range(1,size):
        random[x]=(random[x-1]*a+c)%mod
    final=(random/(mod*1.))
    return final

def Suffle(Deck,n=1,a=2521,c=13,mod=2**16,seed=2*17+7):
    NumberofCards=52
    final=PRNG(n*NumberofCards,2521,13,2**16,seed)
    index=final[(n-1)*NumberofCards:n*NumberofCards].argsort()
    return Deck[index].astype(int)
