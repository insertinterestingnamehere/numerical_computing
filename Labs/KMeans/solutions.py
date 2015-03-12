import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import decomposition
from scipy import linalg as la

def kmeans(data,n_clusters,init='random',max_iter=300,normalize=False):
    n,m = data.shape
    norms = np.zeros((n,n_clusters))
    if init=='random':
        means = data.mean(axis=0)+np.random.randn(n_clusters,m)*data.std()
    else:
        means=init
    if normalize:
        means /= np.linalg.norm(means,axis=1)[:,np.newaxis]
    for i in xrange(n_clusters):
        norms[:,i] = ((data-means[i,:])**2).sum(axis=1)
    labels = np.argmin(norms,axis=1)
    it =0
    for j in xrange(max_iter):
        it += 1
        means = np.array([data[labels==i].mean(axis=0) for i in xrange(n_clusters)])
        if normalize:
            means /= np.linalg.norm(means,axis=1)[:,np.newaxis]
        for i in xrange(n_clusters):
            norms[:,i] = ((data-means[i,:])**2).sum(axis=1)
        new_labels = np.argmin(norms,axis=1)
        if not np.allclose(labels,new_labels):
            labels=new_labels
        else:
            break

    inertia = 0
    for i in xrange(n_clusters):
        inertia += ((data[labels==i]-means[i,:])**2).sum()
    return means,labels,inertia

def clusterIris():    
    iris = load_iris()
    
    X = iris.data
    # pre-process
    Y = X - X.mean(axis=0)
    # get SVD
    U,S,VT = la.svd(Y,full_matrices=False)
    # project onto the first two principal components
    Yhat = U[:,:2].dot(np.diag(S[:2]))
    
    # cluster 10 times, retaining the best
    means = None
    labs = None
    inertia=np.inf
    
    for j in xrange(10):
        m,l,i = kmeans(Yhat,3)
        if i < inertia:
            inertia=i
            means=m
            labs=l
    
    setosa = iris.target==0
    versicolor = iris.target==1
    virginica = iris.target==2
    p1, p2 = Yhat[:,0], Yhat[:,1]
    mrkr = []
    for flower, m,n in zip([setosa,versicolor,virginica],['*','.','^'],['Setosa','Versicolor','Virginica']):
        mr = plt.scatter([],[],color='k',marker=m,label=n)
        mrkr.append(mr)
        for i,c in enumerate(['cyan','red','green']):
            msk = np.where(labs[flower]==i)[0]
            if msk.any():
                plt.scatter(p1[flower][msk],p2[flower][msk], marker=m, color=c)
    mrkr.append(plt.scatter(means[:,0],means[:,1],marker='+',s=100,linewidths=2,label="Means"))
    
    plt.legend(handles=mrkr, loc=2)
    plt.ylim([-4,5])
    plt.xlim([-4,4])
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

def loadEarthquakes(path):
    latitudes = []
    longitudes = []
    for i in xrange(1,7):
        with open(path.format(i),'r') as f:
            for line in f:
                la = float(line[20:25])/1000
                s = line[25]=='S'
                lo = float(line[26:32])/1000
                w = line[32]=='W'
                latitudes.append(la*(-1)**s)
                longitudes.append(lo*(-1)**w)
    latitudes=np.array(latitudes)
    longitudes = np.array(longitudes)
    return latitudes,longitudes

def sphericalToEuclidean(r, theta, phi):
    """
    theta and phi must be in radians!!
    """
    eucl = np.zeros((len(theta),3))
    eucl[:,0] = np.sin(phi)*np.cos(theta)
    eucl[:,1] = np.sin(phi)*np.sin(theta)
    eucl[:,2] = np.cos(phi)
    return r*eucl
def euclideanToSpherical(pts):
    """
    returns answers in radians.
    """
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    phi=np.arccos(z)
    theta=np.arctan2(y,x)
    return phi, theta
    
def clusterEarthquakes(path):
    # load in the data
    latitudes, longitudes = loadEarthquakes(path)
    # convert to euclidean coordinates
    eucl = sphericalToEuclidean(1.,longitudes*np.pi/180,(90-latitudes)*np.pi/180)
    # cluster 10 times, retaining best
    best_i = np.inf
    best_m = None
    best_l = None
    for j in xrange(10):
        m,l,i = kmeans(eucl,15,normalize=True)
        if i < best_i:
            best_i=i
            best_m = m
            best_l = l
    # get means back into latitude and longitude coordinates
    m_phi, m_longitude = euclideanToSpherical(best_m)
    m_latitude = np.pi/2-m_phi
    return latitudes, longitudes, m_latitude, m_longitude, best_l
    
def plotClusters(lats, longs, m_lats, m_longs, labs):
    for i in xrange(15):
        msk=labs==i
        c=np.random.rand(3)
        plt.scatter(longs[msk],lats[msk],marker='.',
                    color=c)
    plt.scatter(m_longs*180/np.pi,m_lats*180/np.pi,marker='+',s=100,linewidths=2,color='k')    
    plt.show()
