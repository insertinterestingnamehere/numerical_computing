import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model 


housing = np.load('housingprices.npy')
challenger = np.load('challenger.npy')

def raw():
    plt.plot(housing[:,1], housing[:,0], 'o')
    plt.savefig("california.pdf")
    plt.clf()

def linear():
    X=np.ones((42,2))
    X[:,1]=housing[:,1]
    Y = housing[:,0]
    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    plt.plot(X[:,1],Y,'o')
    xseq=np.arange(0,12,.1)
    plt.plot(xseq,betahat[0]+betahat[1]*xseq)
    plt.savefig("cali-linear.pdf")
    plt.clf()


def cubic():
    X=np.ones((42,4))
    X[:,1]=housing[:,1]
    X[:,2]=X[:,1]**2
    X[:,3]=X[:,1]**3
    Y = housing[:,0]

    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    plt.plot(X[:,1],Y,'o')
    xseq=np.arange(0,12,.1)
    plt.plot(xseq,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3)
    plt.savefig("cali-quadratic.pdf")
    plt.clf()

def quartic():
    X=np.ones((42,5))
    X[:,1]=housing[:,1]
    X[:,2]=X[:,1]**2
    X[:,3]=X[:,1]**3
    X[:,4]=X[:,1]**4
    Y=housing[:,0]

    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    plt.plot(X[:,1],Y,'o')
    xseq=np.arange(0,12,.1)
    plt.plot(xseq,betahat[0]+betahat[1]*xseq
            +betahat[2]*xseq**2+betahat[3]*xseq**3+betahat[4]*xseq**4)
    plt.ylim([0,600000])
    plt.savefig("cali-quartic.pdf")
    
    
    
def challenger_cubic():
    plt.plot(challenger[:,0], challenger[:,1], 'o')
    plt.xlim(30,100)
    plt.xlabel('Ambient Temperature (F)')
    plt.ylim(-0.5,1.5)
    plt.ylabel('O-ring Damage Present')
    plt.title('Potential for Shuttle Damage - With Cubic Approximation')
    X=np.ones((challenger.shape[0],4))
    X[:,1] = challenger[:,0]
    Y=challenger[:,1]
    X[:,2]=X[:,1]**2
    X[:,3]=X[:,1]**3
    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    xseq=np.arange(30,100,.5)
    plt.plot(xseq,betahat[0]+betahat[1]*xseq+betahat[2]*xseq**2+betahat[3]*xseq**3)
    plt.savefig('cubicthrulogitpoints.pdf')
    plt.clf()

def challenger_logistic():
    ###Logreg plot #2
    plt.plot(challenger[:,0], challenger[:,1],'o')
    plt.xlim(30,100)
    plt.xlabel('Ambient Temperature (F)')
    plt.ylim(-0.5,1.5)
    plt.ylabel('O-ring Damage Present')
    plt.title('Potential for Shuttle Damage - With Logistic Regression Prediction')
    #X=np.ones((dat.shape[0],2))
    #X[:,1]=dat[:,0]
    X=challenger[:,0].reshape((23,1))
    Y=challenger[:,1]
    logreg = linear_model.LogisticRegression(C=1000000,penalty="l2")
    logreg.fit(X,Y)
    coef=logreg.coef_[0]
    xseq=np.arange(30,100,.5)[:,np.newaxis]
    #xseqmat=np.ones((len(xseq),2))
    #xseqmat[:,1]=xseq
    xB=logreg.intercept_[0]+logreg.coef_[0][0]*xseq
    #plt.plot(xseq,1/(np.exp(-xB)+1))
    plt.plot(xseq,logreg.predict_proba(xseq)[:,1])
    plt.savefig("logreg.pdf")
    plt.clf()

if __name__ == "__main__":
    raw()
    linear()
    cubic()
    quartic()
    challenger_cubic()
    challenger_logistic()
    