import scipy as sp
from scipy import sparse as spar
from scipy . sparse import linalg as sparla
from scipy import linalg as la
import numpy as np
from scipy import eye
from math import sqrt
import math
import matplotlib . pyplot as plt
from collections import Counter
import fractions

def PRNG(size,a=1103515245,c=12345,mod=2**31-1,seed=4329):
    x1=seed
    for x in range(43):
        x1=(x1*a+c)%mod
    random=sp.zeros(size)
    random[0]=(x1*a+c)%mod
    for x in range(1,size):
        random[x]=(random[x-1]*a+c)%mod
    final=(random/(mod*1.))
    return final

def PRNGint(size,least=0,greatest=2,a=1103515245,c=12345,mod=2**31-1,seed=432946458):
    final=PRNG(size,a,c,mod,seed)
    final=(final*(greatest-least)).astype(int)+least
    return final

n=512
max=2
final=PRNGint(n**2,0,max)#,25214903917,11,2**48,4)
plt.imshow(final.reshape(n,n))
plt.gray()
#plt.jet()
plt.show()

n=2**9
max=52
final=PRNG(n**2,25214903917,11,2**48,2*17+7)
plt.imshow(final.reshape(n,n))
#plt.gray()
plt.jet()
plt.show()