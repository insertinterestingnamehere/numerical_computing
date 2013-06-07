def shuffle(deck):
    size = len(deck)
    newdeck=np.empty_like(deck)
    for i in xrange(size/2):
         num=random.randint(0,1)
         if num==0:
            newdeck[i*2]=deck[i]
            newdeck[i*2+1]=deck[size/2+i]
         else:
            newdeck[i*2]=deck[size/2+i]
            newdeck[i*2+1]=deck[i]
    deck=newdeck.copy()
    return deck

def shuffleB(deck):
    s=len(deck)/2
    ra=np.random.randint(0,2,s)
    tempdeck=deck[range(s)+(1-ra)*s].copy()
    deck[::2]=deck[range(s)+ra*s]
    deck[1::2]=tempdeck
    return deck

def imageEditor(X,edit,n=1):
    if edit==1:
        Xnew=255-X
        
    elif edit==2:
        Xnew=255-np.tile(((X/3).sum(2)).reshape(X.shape[0],X.shape[1],1),(1,1,3))
        
    else:
        Xnew=X.copy()/n
        for i in range(1,n):
            Xnew[:,:-i,:]=Xnew[:,:-i,:]+X[:,i:,:]/n
            Xnew[:,-i:,:]=Xnew[:,-i:,:]+Xnew[:,-i:,:]/n
            
    plt.imshow(Xnew)
    plt.show()


Xnew=X.copy()
Xnew=np.where(X<10,245,X)
plt.imshow(Xnew)
plt.show()