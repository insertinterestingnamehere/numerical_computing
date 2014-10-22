import scipy as sp
import time
from bjCommon import *

def PlayBlackJack(numberOfRounds=1):

    DeckArray = sp.array(Deck)
    values = [11,2,3,4,5,6,7,8,9,10,10,10,10]*4
    breaker=[]
    wins=sp.array([0,0])
    seed=int(time.time())
    
    allDecks = Suffle(numberOfRounds,25214903917,11,2**48,seed)
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

def totalValue(hand,Deck):
    values = [11,2,3,4,5,6,7,8,9,10,10,10,10]*4
    sum=sp.array([0,0])
    for i in hand:
        if values[Deck.index(i)]==11:
            sum[0]=sum[0]+11
            sum[1]=sum[1]+1
        else:
            sum[0]=sum[0]+values[Deck.index(i)]
    while sum[0]>21 and sum[1]>0:
        sum[0]=sum[0]-10
        sum[1]=sum[1]-1
    return sum[0]


if __name__ == "__main__":
    import sys
    if(len(sys.argv) < 2):
        print "Please Specify the Number of Games"
    else:
        PlayBlackJack(int(sys.argv[1]))
