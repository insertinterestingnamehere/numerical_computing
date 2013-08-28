import numpy as np
import matplotlib.pyplot as plt

dataset = np.load('housingprices.npy')
plt.plot(dataset[:,0],dataset[:,1],'o')
plt.show()


#simple linear
X=np.ones((42,2))
X[:,1]=dataset[:,1]
Y=dataset[:,0]

betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
plt.plot(X[:,1],Y,'o')
xseq=np.arange(0,12,.1)
plt.plot(xseq,betahat[0]+betahat[1]*xseq)
plt.show()




#cubic
X=np.ones((42,4))
X[:,1]=dataset[:,1]
X[:,2]=X[:,1]**2
X[:,3]=X[:,1]**3
Y=dataset[:,0]

betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
plt.plot(X[:,1],Y,'o')
xseq=np.arange(0,12,.1)
plt.plot(xseq,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3)
plt.show()



#cubic
X=np.ones((42,5))
X[:,1]=dataset[:,1]
X[:,2]=X[:,1]**2
X[:,3]=X[:,1]**3
X[:,4]=X[:,1]**4
Y=dataset[:,0]

betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
plt.plot(X[:,1],Y,'o')
xseq=np.arange(0,12,.1)
plt.plot(xseq,betahat[0]+betahat[1]*xseq
         +betahat[2]*xseq**2+betahat[3]*xseq**3+betahat[4]*xseq**4)
plt.ylim([0,600000])
plt.show()
