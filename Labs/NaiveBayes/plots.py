import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn import  datasets

def boundaries():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2] 
    y = iris.target    
    h = .02
    means = np.empty((X.shape[1], len(set(y))))
    for i,lab in enumerate(list(set(y))):
        means[:,i] = X[y==lab].mean(axis=0)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    nb = GaussianNB()
    nb.fit(X, y)
    Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(means[0,:], means[1,:])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig("decision_boundary.pdf")
    plt.clf()
    
if __name__ == "__main__":
    boundaries()
