import numpy as np
import matplotlib.pyplot as plt
import pickle
import gmm
from scipy import stats
from scipy import optimize as opt
import scipy as sp

infile = open('homicides','r')
data = pickle.load(infile)
infile.close()

model = gmm.GMM(3)
model.train(data,random=False)

mins = np.min(data,0)
maxes = np.max(data,0)
x = np.arange(mins[0]-.01,maxes[0]+.01,.0005)
y = np.arange(mins[1]-.01,maxes[1]+.01,.0005)
X,Y = np.meshgrid(x,y)
gmmimmat = np.zeros(X.shape)

for i in xrange(X.shape[0]):
	print i
	for j in xrange(X.shape[1]):
		gmmimmat[i,j] = model.dgmm(np.array([X[i,j],Y[i,j]]))

plt.imshow(gmmimmat,origin='lower')
plt.ylim([0,X.shape[0]])
plt.xlim([0,X.shape[1]])
plt.show()

kdeimmat = np.zeros(X.shape)
kernel = stats.gaussian_kde(data.T)
for i in xrange(X.shape[0]):
	print i
	for j in xrange(X.shape[1]):
		kdeimmat[i,j] = kernel.evaluate(np.array([X[i,j],Y[i,j]]))

plt.imshow(kdeimmat,origin='lower')
plt.ylim([0,X.shape[0]])
plt.xlim([0,X.shape[1]])
plt.show()

def fgmm(x):
	return abs(np.sum(gmmimmat[gmmimmat>x])*.0005**2 - 0.95)

thresh = opt.fmin(fgmm,10)[0]
bools = gmmimmat > thresh

mat = np.zeros(X.shape)
mat += bools
plt.imshow(mat,origin='lower')
plt.ylim([0,X.shape[0]])
plt.xlim([0,X.shape[1]])
plt.show()

def fkde(x):
	return abs(np.sum(kdeimmat[kdeimmat>x])*.0005**2 - 0.95)

thresh = opt.fmin(fkde,10)[0]
bools = kdeimmat > thresh

mat = np.zeros(X.shape)
mat += bools
plt.imshow(mat,origin='lower')
plt.ylim([0,X.shape[0]])
plt.xlim([0,X.shape[1]])
plt.show()
