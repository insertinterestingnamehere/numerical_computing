import numpy as np
import matplotlib.pyplot as plt

raw=open('housing.dat','rb').read().split('\n')
raw=[x.split('\t') for x in raw]
raw=raw[1:-1]

dataset = raw[42:]
dataset = [[float(x[1]),float(x[3])] for x in dataset]
dataset = np.array(dataset)

states = [x[0] for x in raw]
states = sorted(list(set(states)))

plt.plot(dataset[:,0],dataset[:,1],'o')
plt.show()

for i in xrange(51): plt.plot(dataset[(i*42):(42*(i+1)-1),0],dataset[(i*42):(42*(i+1)-1),1])

i=4 #california
plt.plot(dataset[(i*42):(42*(i+1)-1),0],dataset[(i*42):(42*(i+1)-1),1])
plt.plot(dataset[(i*42):(42*(i+1)-1),0],dataset[(i*42):(42*(i+1)-1),1],'o')
plt.show()

#simple linear
X=np.ones((42,2))
X[:,1]=dataset[(i*42):(42*(i+1)),0]
Y=dataset[(i*42):(42*(i+1)),1]

betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
plt.plot(X[:,1],Y,'o')
xseq=np.arange(2000,2012,.1)
plt.plot(xseq,betahat[0]+betahat[1]*xseq)
plt.show()

#quadratic
X=np.ones((42,3))
X[:,1]=dataset[(i*42):(42*(i+1)),0]
X[:,2]=X[:,1]**2
Y=dataset[(i*42):(42*(i+1)),1]

betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
plt.plot(X[:,1],Y,'o')
xseq=np.arange(2000,2012,.1)
plt.plot(xseq,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2)
plt.show()



#cubic
X=np.ones((42,2))
X[:,1]=dataset[(i*42):(42*(i+1)),0] - 2000
X[:,2]=X[:,1]**2
X[:,3]=X[:,1]**3
Y=dataset[(i*42):(42*(i+1)),1]

betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
plt.plot(X[:,1],Y,'o')
xseq=np.arange(3,12,.1)
plt.plot(xseq,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3)
plt.show()



#### plot 1
xx=np.random.normal(5,1.5,20)
yy=np.random.normal(xx,1)
plt.plot(xx,yy,'o')
plt.xlim([0,10])
plt.ylim([0,10])

xmat=np.ones((len(xx),2))
xmat[:,1]=xx
betahat = np.linalg.inv(xmat.T.dot(xmat)).dot(xmat.T.dot(yy))

plotx=np.arange(0,11)
plt.plot(plotx,betahat[0]+betahat[1]*plotx,'r-')
plt.show()


###plot 2
plt.plot(dataset[:,1]+2000,dataset[:,0],'o')
plt.ylim([0,600000])
plt.title('California Median House Price')
plt.show()

###plot 3 - logit
plotp = np.arange(.00001,1,.00001)
plotlogit = np.log(plotp/(1-plotp))
plt.plot(plotp,plotlogit)
plt.xlim([-.01,1.01])
plt.ylim([-10,10])
plt.xlabel('x')
plt.ylabel('logit(x)')
plt.show()

###plot 4 - logistic regression
xxx = np.arange(-7,8,1)
yyy = np.array([0,0,0,0,1,0,0,1,0,1, 1,1,1,1,1])
plotp = np.arange(.001,1,.001)
plotlogit = np.log(plotp/(1-plotp))
plt.plot(plotlogit,plotp)
plt.plot(xxx,yyy,'o')
plt.xlabel('X')
plt.ylabel('Probability of Event (Y=1)')
plt.show()


########################## makes defaulting data for logistic regression.
logitdat = np.ones((80,5))
logitdat[:,1]=np.random.poisson(8,80)*50000 + np.random.poisson(20,80)*1000 #loan size
    #np.median(logitdat[:,1])
logitdat[:,2]=np.random.poisson(10,80)*10000 + np.random.poisson(10,80)*1000 #income
logitdat[:,3]=np.random.binomial(1,.3,80) #marital status
logitdat[:,4]=10*(np.random.poisson(30,80)+np.random.poisson(30,80)) #credit score
xb=(1/100000.)*logitdat[:,1]-3*logitdat[:,3]- (1/100.)*logitdat[:,4]
ppp = np.exp(xb)/(1+np.exp(xb))
logitdat[:,0] = np.random.binomial(1,ppp)
np.save('defaulting.npy',logitdat)
#########################

logitdat=np.load('defaulting.npy')
Y=logitdat[:,0]
X=logitdat[:,1:]

from sklearn import linear_model 
logreg = linear_model.LogisticRegression()
logreg.fit(X,Y)


##########################################
#Challenger data
bob=open('challenger.txt').read().split('\r\n')
bob=[x.split('\t') for x in bob]
bob=bob[1:-1]
bob=[[float(y) for y in x] for x in bob]
bob=np.array(bob)
bob[17,1]=1
bob[0,1]=1
np.save('challenger.npy',bob)


###Logreg plot #1
dat = np.load('challenger.npy')
plt.plot(dat[:,0],dat[:,1],'o')
plt.xlim(30,100)
plt.xlabel('Ambient Temperature (F)')
plt.ylim(-0.5,1.5)
plt.ylabel('O-ring Damage Present')
plt.title('Potential for Shuttle Damage - With Cubic Approximation')
X=np.ones((dat.shape[0],4))
X[:,1]=dat[:,0]
Y=dat[:,1]
X[:,2]=X[:,1]**2
X[:,3]=X[:,1]**3
betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
xseq=np.arange(30,100,.5)
plt.plot(xseq,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3)
plt.show()


###Logreg plot #2
plt.plot(dat[:,0],dat[:,1],'o')
plt.xlim(30,100)
plt.xlabel('Ambient Temperature (F)')
plt.ylim(-0.5,1.5)
plt.ylabel('O-ring Damage Present')
plt.title('Potential for Shuttle Damage - With Logistic Regression Prediction')
#X=np.ones((dat.shape[0],2))
#X[:,1]=dat[:,0]
X=dat[:,0].reshape([23,1])
Y=dat[:,1]#.reshape([23,1])
from sklearn import linear_model 
logreg = linear_model.LogisticRegression(C=1000000,penalty="l2")
logreg.fit(X,Y)
coef=logreg.coef_[0]
xseq=np.arange(30,100,.5)
#xseqmat=np.ones((len(xseq),2))
#xseqmat[:,1]=xseq
xB=logreg.intercept_[0]+logreg.coef_[0][0]*xseq
#plt.plot(xseq,1/(np.exp(-xB)+1))
plt.plot(xseq,logreg.predict_proba(xseq2)[:,1])
plt.show()

xB = logreg.coef_*31 + logreg.intercept_
1/(np.exp(-xB)+1)






