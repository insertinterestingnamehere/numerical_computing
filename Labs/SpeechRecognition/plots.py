import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from speechrecognition import sample_gmmhmm

def mixture():
    M = 4
    weights = np.random.dirichlet(np.ones(M),size=1)[0]
    means = np.linspace(-5,5,M) + np.random.randn(M)*.1
    sigs = np.random.random(size=M)+1
    dom = np.linspace(-10,10,500)
    curve = np.zeros(500)
    for i in xrange(M):
        #plt.plot(dom, st.norm.pdf(dom, loc=means[i], scale=np.sqrt(sigs[i])))
        curve += weights[i]*st.norm.pdf(dom, loc=means[i], scale=np.sqrt(sigs[i]))
    plt.plot(dom,curve)
    plt.savefig("mixture.pdf")
    plt.clf()

def samples():
    A = np.array([[.65, .35], [.15, .85]]).T
    pi = np.array([.8, .2])
    weights = np.array([[1.], [1.]])
    means1 = np.array([[0., 17.]])
    means2 = np.array([[-12., -2.]])
    means = np.array([means1, means2])
    covars1 = np.array([np.eye(2)*5])
    covars2 = np.array([np.eye(2)*10])
    covars = np.array([covars1, covars2])
    gmmhmm = [A, weights, means, covars, pi]
    states, obs = sample_gmmhmm(gmmhmm,10)
    plt.plot(obs[:,0], obs[:,1], "y--")
    plt.scatter(obs[states==0,0], obs[states==0,1], c='b', s=40)
    plt.scatter(obs[states==1,0], obs[states==1,1], c='r', s=40)
    plt.savefig("samples.pdf")
    plt.clf()

if __name__ == "__main__":
    samples()
