import matplotlib.pyplot as plt
import numpy as np

###Problem 1
(180+5)/(600+5.)



N=10000
mathematics = np.random.normal(500,100,N)
reading = np.random.normal(500,100,N)
writing = np.random.normal(500,100,N)
SATscore = mathematics + reading + writing

plt.hist(SATscore)
plt.show()

np.std(SATscore)
np.sqrt(3*10000)
np.var(SATscore)
np.mean(SATscore)

#problem 2
sig = np.ones((3,3)) *5000
sig[0,0]=10000
sig[1,1]=10000
sig[2,2]=10000
sig[1,2]=7000
sig[2,1]=7000

mvnscores = np.random.multivariate_normal(mu,sig,10000)
SATscores = np.sum(mvnscores,axis=1)

plt.hist(SATscores,bins=100)
plt.show()

np.mean(SATscores)
np.var(SATscores)
np.std(SATscores)
np.sqrt(64000)
