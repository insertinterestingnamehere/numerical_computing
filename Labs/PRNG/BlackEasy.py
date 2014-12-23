import scipy as sp
import time
import random

from bjCommon import *

def PlayBlackJack(numberOfRounds=1):
    
    DeckArray = sp.array(Deck)
    values = [11,2,3,4,5,6,7,8,9,10,10,10,10]*4
    breaker=[]
    wins=sp.array([0,0])
    seed=random.randrange(10000000)
    allDecks = shuffle(numberOfRounds,2521,13,2**16,seed)
    for x in range(numberOfRounds):
        PlayDeck= convertToName(allDecks[x]).tolist()
        dealerHand=[]
        playerHand=[]
        playerHand.append(PlayDeck.pop(0))
        dealerHand.append(PlayDeck.pop(0))
        playerHand.append(PlayDeck.pop(0))
        dealerHand.append(PlayDeck.pop(0))
        print 'Your Hand:',playerHand, totalValue(playerHand, Deck)
        print 'Dealer\'s Hand:',dealerHand[0]
        val=raw_input("hit or stand? ")
        while val=='hit':
            playerHand.append(PlayDeck.pop(0))
            print '\nYour Hand:',playerHand, totalValue(playerHand, Deck)
            print 'Dealer\'s Hand:',dealerHand[0]
            if totalValue(playerHand, Deck)<22:
                val=raw_input("hit or stand? ")
            else:
                val='stand'
        while totalValue(dealerHand,Deck)<17:
            dealerHand.append(PlayDeck.pop(0))
        print dealerHand
        if totalValue(playerHand,Deck)>21:
            wins[1]=wins[1]+1
        elif totalValue(dealerHand,Deck)>21:
            wins[0]=wins[0]+1
        elif totalValue(playerHand,Deck)>totalValue(dealerHand,Deck):
            wins[0]=wins[0]+1
        elif totalValue(playerHand,Deck)<totalValue(dealerHand,Deck):
            wins[1]=wins[1]+1
        print wins
        print
    return 0


if __name__ == "__main__":
    import sys
    if(len(sys.argv) < 2):
        print "Please Specify the Number of Games"
    else:
        PlayBlackJack(int(sys.argv[1]))
