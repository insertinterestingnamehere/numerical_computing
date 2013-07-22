import scipy as sp
import time

def PlayBlackJack(numberOfRounds=1):
    hearts=['Aheart','2heart','3heart','4heart','5heart','6heart','7heart','8heart','9heart','10heart','Jheart','Qheart','Kheart']
    spades=['Aspade','2spade','3spade','4spade','5spade','6spade','7spade','8spade','9spade','10spade','Jspade','Qspade','Kspade']
    clubs=['Aclub','2club','3club','4club','5club','6club','7club','8club','9club','10club','Jclub','Qclub','Kclub']
    diamonds=['Adiamond','2diamond','3diamond','4diamond','5diamond','6diamond','7diamond','8diamond','9diamond','10diamond','Jdiamond','Qdiamond','Kdiamond']
    Deck=hearts+diamonds+clubs+spades
    DeckArray = sp.array(Deck)
    values = [11,2,3,4,5,6,7,8,9,10,10,10,10]*4
    breaker=[]
    wins=sp.array([0,0])
    seed=int(time.time())
    for x in range(numberOfRounds):
        PlayDeck=Suffle(DeckArray,(x+1)*((seed%50+(seed+1)%2)),seed).tolist()
        dealerHand=[]
        playerHand=[]
        playerHand.append(PlayDeck.pop())
        dealerHand.append(PlayDeck.pop())
        playerHand.append(PlayDeck.pop())
        dealerHand.append(PlayDeck.pop())
        print 'Your Hand:',playerHand, totalValue(playerHand, Deck)
        print 'Dealer\'s Hand:',dealerHand[0]
        val=raw_input("hit or stand? ")
        while val=='hit':
            playerHand.append(PlayDeck.pop())
            print '\nYour Hand:',playerHand, totalValue(playerHand, Deck)
            print 'Dealer\'s Hand:',dealerHand[0]
            if totalValue(playerHand, Deck)<22:
                val=raw_input("hit or stand? ")
            else:
                val='stand'
        while totalValue(dealerHand,Deck)<17:
            dealerHand.append(PlayDeck.pop())
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
            
def PRNG(size,a=2521,c=13,mod=2**16,seed=2*17+7):
    x1=seed
    for x in range(43):
        x1=(x1*a+c)%mod
    random=sp.zeros(size)
    random[0]=(x1*a+c)%mod
    for x in range(1,size):
        random[x]=(random[x-1]*a+c)%mod
    final=(random/(mod*1.))
    return final

def Suffle(Deck,n=1,seed=2*17+7):
    NumberofCards=52
    final=PRNG(n*NumberofCards,2521,13,2**16,seed)
    index=final[(n-1)*NumberofCards:n*NumberofCards].argsort()
    return Deck[index]

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
    PlayBlackJack(int(sys.argv[1]))