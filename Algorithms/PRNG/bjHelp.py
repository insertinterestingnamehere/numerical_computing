from collections import Counter
import scipy as sp
import numpy as np

hearts=['Aheart','2heart','3heart','4heart','5heart','6heart','7heart','8heart','9heart','10heart','Jheart','Qheart','Kheart']
spades=['Aspade','2spade','3spade','4spade','5spade','6spade','7spade','8spade','9spade','10spade','Jspade','Qspade','Kspade']
clubs=['Aclub','2club','3club','4club','5club','6club','7club','8club','9club','10club','Jclub','Qclub','Kclub']
diamonds=['Adiamond','2diamond','3diamond','4diamond','5diamond','6diamond','7diamond','8diamond','9diamond','10diamond','Jdiamond','Qdiamond','Kdiamond']
Deck=hearts+diamonds+clubs+spades

#convertToNum takes in the names of the cards and prints out their original indexes
def convertToNum(breaker):
    return np.array([Deck.index(name) for name in breaker])

#convertToName takes in the indicies of the cards in Deck and prints out their names
def convertToName(breaker,rounds=1,cardNum=3):
    return [Deck[(int)(index)] for index in breaker]

#psuedorandom number genrator that blackjack uses

def PRNG(size,a=25214903917,c=11,mod=2**48,seed=2*17+7):
    x1=seed
    for x in range(43):
        x1=(x1*a+c)%mod
    random=sp.zeros(size)
    random[0]=(x1*a+c)%mod
    for x in range(1,size):
        random[x]=(random[x-1]*a+c)%mod
    final=(random/(mod*1.))
    return final

#prints out the n ordering of cards given a seed value

def Suffle(n=1,a=25214903917,c=11,mod=2**48,seed=1):
    NumberofCards=52
    final=PRNG(n*NumberofCards,a,c,mod,seed)
    index=sp.zeros((n,NumberofCards))
    for x in range(n):
        index[x,:]=final[(x)*NumberofCards:(x+1)*NumberofCards].argsort()
    return sp.fliplr(index).astype(int)

