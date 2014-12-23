import matplotlib.pyplot as plt
import numpy as np

###Problem 1


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





#Figures:
plt.subplot(311)
dat=np.random.exponential(1,[10000,10])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,2.5)
plt.ylim(0,400)
plt.title('n=10')

plt.subplot(312)
dat=np.random.exponential(1,[10000,20])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,2.5)
plt.ylim(0,400)
plt.title('n=20')

plt.subplot(313)
dat=np.random.exponential(1,[10000,30])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,2.5)
plt.ylim(0,400)
plt.title('n=30')

plt.show()



#poisson
xmax=6
ymax=800
plt.subplot(311)
dat=np.random.poisson(3,[10000,10])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.title('n=10')

plt.subplot(312)
dat=np.random.poisson(3,[10000,20])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.title('n=20')

plt.subplot(313)
dat=np.random.poisson(3,[10000,30])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.title('n=30')

plt.show()


#beta
xmax=1
ymax=400
plt.subplot(311)
dat=np.random.beta(.1,.1,[10000,10])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.title('n=10')
#
plt.subplot(312)
dat=np.random.beta(.1,.1,[10000,20])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.title('n=20')
#
plt.subplot(313)
dat=np.random.beta(.1,.1,[10000,30])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.title('n=30')
plt.show()



#beta
xmin=12
xmax=20
ymax=700
plt.subplot(311)
dat=np.random.binomial(20,.8,[10000,10])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(xmin,xmax)
plt.ylim(0,ymax)
plt.title('n=10')
#
plt.subplot(312)
dat=np.random.binomial(20,.8,[10000,20])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(xmin,xmax)
plt.ylim(0,ymax)
plt.title('n=20')
#
plt.subplot(313)
dat=np.random.binomial(20,.8,[10000,30])
plt.hist(np.mean(dat,1),bins=100)
plt.xlim(xmin,xmax)
plt.ylim(0,ymax)
plt.title('n=30')
plt.show()
