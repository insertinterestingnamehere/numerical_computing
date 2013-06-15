import numpy as np
import timeit
import matplotlib . pyplot as plt
import math

def naiveKS(items,maxim):
    currentBest =[]
    currentHigh = 0
    temp = []
    for x in powerset(items):
        weight,value = checker(x)
        
        if weight <= maxim and value > currentHigh:
            currentBest=x[:]
            currentHigh=value
    return currentBest,currentHigh

def checker(items):
    weight = 0
    value = 0
    for x in items:
        weight = weight + x[1]
        value = value + x[0]
    return weight,value

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def betterKS(items,W):
    n=len(items)
    m=sp.zeros((n+1,W+1))
    currentBest =[]

    for i in range(1,n+1):
        for j in range(W+1):
            if j >= items[i-1][1]:
                m[i, j] = max(m[i-1, j], m[i-1, j-items[i-1][1]] + items[i-1][0])
                
            else:
                m[i, j] = m[i-1, j]
                
    i=n
    j=W
    while m[i,j]!=0:
        if m[i,j]!=m[i-1,j]:
            currentBest.append(items[i-1])
            j=j-items[i-1][1]
        i=i-1
            
    return currentBest

def ZKS(items,W):
    n=len(items)
    currentBest =dict()
    before=dict()
    for j in range(W+1):
        before[j]=[0]
    
    for i in range(1,n+1):
        for j in range(W+1):
            if j >= items[i-1][1] and (before[j-items[i-1][1]][-1] + items[i-1][0]>before[j][-1]):
                
                currentBest[j] = before[j-items[i-1][1]][0:-1]+[items[i-1]] +[before[j-items[i-1][1]][-1] + items[i-1][0]]
                
            else:
                currentBest[j]= before[j]
        before=currentBest.copy()
        return currentBest[W][0:-1]

def timeFun(f,*args,**kargs):
    pfunc = lambda: f(*args, **kargs)
    theTime=timeit.Timer(pfunc)
    return min(theTime.repeat(2,1))
        
    

numI=[]
times=[]
for x in range(10):
    num=11+x
    q=1000
    numI.append(num)
    items=[]
    for x in range(num):
        items.append((sp.random.rand()*100,sp.random.randint(1,q)))
    times.append(timeFun(naiveKS,items,q*10))

plt.plot(numI,times)
plt.show()

weights=[]
times=[]
for x in range(10):
    num=15
    q=1000*(x+1)
    weights.append(q)
    items=[]
    for x in range(num):
        items.append((sp.random.rand()*100,sp.random.randint(1,q)))
    times.append(timeFun(naiveKS,items,q*10))

plt.plot(weights,times)
plt.show()

numofI=[]
times=[]
times1=[]
for x in range(10):
    num=(x+1)+10
    numofI.append(num)
    q=1000
    items=[]
    for x in range(num):
        items.append((sp.random.rand()*q,sp.random.randint(1,q)))
    times.append(timeFun(ZKS,items,q*10))
    times1.append(timeFun(betterKS,items,q*10))

plt.plot(numofI,np.array([times1,times]).T)
plt.show()

weights=[]
times=[]
times1=[]
for x in range(10):
    num=15
    q=1000*(x+1)
    weights.append(q)
    items=[]
    for x in range(num):
        items.append((sp.random.rand()*100,sp.random.randint(1,q)))
    times.append(timeFun(ZKS,items,q*10))
    times1.append(timeFun(betterKS,items,q*10))

plt.plot(weights,np.array([times1,times]).T)
plt.show()