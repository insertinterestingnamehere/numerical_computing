#Solutions

###plot 5 - cubic fit
plt.plot(dataset[:,1]+2000,dataset[:,0],'o')
plt.ylim([0,600000])
plt.title('California Median House Price - With Quadratic Approximation')
X=np.ones((42,4))
X[:,1]=dataset[:,1]
X[:,2]=dataset[:,1]**2
X[:,3]=dataset[:,1]**3
Y=dataset[:,0]
betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
xseq=np.arange(0,12,.1)
plt.plot(xseq+2000,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3)
plt.show()


###plot 6 - quartic fit
plt.plot(dataset[:,1]+2000,dataset[:,0],'o')
plt.ylim([0,600000])
plt.title('California Median House Price - With Quadratic Approximation')
X=np.ones((42,5))
X[:,1]=dataset[:,1]
X[:,2]=dataset[:,1]**2
X[:,3]=dataset[:,1]**3
X[:,4]=dataset[:,1]**4
Y=dataset[:,0]
betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
xseq=np.arange(0,12,.1)
plt.plot(xseq+2000,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3+betahat[4]*xseq**4)
plt.show()


##logistic regression
dat = np.load('challenger.npy')

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
