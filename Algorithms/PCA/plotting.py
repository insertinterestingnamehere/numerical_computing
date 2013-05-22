import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import scipy as sp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition

iris = datasets.load_iris()

def iris_base():
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,aspect='equal')
    plt.plot(iris.data[50:150,0],iris.data[50:150,2],'k.')
    plt.xlim([2,8])
    plt.ylim([2,8])
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")

    return fig

def ibase():
    iris_base()
    plt.savefig('iris0.pdf')

def iris1():
    fig = iris_base()
    pca = decomposition.PCA(n_components=2)
    pca.fit(iris.data[50:150,sp.array([0,2])])
    mean = sp.mean(iris.data[50:150,sp.array([0,2])],0)
    stds = sp.std(iris.data[50:150,sp.array([0,2])],0)
    components = pca.components_

    plt.quiver(mean[0],mean[1],1.5*stds[0],0,scale_units='xy',angles='xy',scale=1)
    plt.quiver(mean[0],mean[1],0,1.5*stds[1],scale_units='xy',angles='xy',scale=1)
    plt.savefig('iris1.pdf')

def iris2():
    fig = iris_base()
    pca = decomposition.PCA(n_components=2)
    pca.fit(iris.data[50:150,sp.array([0,2])])
    mean = sp.mean(iris.data[50:150,sp.array([0,2])],0)
    stds = sp.std(iris.data[50:150,sp.array([0,2])],0)
    components = pca.components_
    variance_ratio = pca.explained_variance_ratio_

    plt.quiver(mean[0],mean[1],-2*variance_ratio[0]*components[0,0],-2*variance_ratio[0]*components[0,1],scale_units='xy',angles='xy',scale=1)
    plt.quiver(mean[0],mean[1],5*variance_ratio[1]*components[1,0],5*variance_ratio[1]*components[1,1],scale_units='xy',angles='xy',scale=1)
    plt.savefig('iris2.pdf')

ibase()
iris1()
iris2()