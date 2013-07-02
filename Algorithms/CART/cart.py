import scipy as sp
import numpy as np
import random
from scipy import stats
import pickle

def partitiondata(filename_1,filename_2):
    infile = open(filename_1,'r')
    data = pickle.load(infile)
    infile.close()
    infile = open(filename_2,'r')
    target = pickle.load(infile)
    infile.close()
    N = len(target)
    inds = random.sample(sp.arange(0,N),N)
    trainingdata = [data[i] for i in inds[0:800]]
    trainingtarget = sp.array([target[i] for i in inds[0:800]])
    testdata = [data[i] for i in inds[800:]]
    testtarget = sp.array([target[i] for i in inds[800:]])
    return trainingdata,trainingtarget,testdata,testtarget

def gini(y,classes):
	N = len(y)
	fks = sp.array([sum(y==classes[i]) for i in xrange(len(classes))])/float(N)
	return 1 - sp.sum(fks**2)

class Node(object):

	def __init__(self,data,y,classes,variables,types,depth=1,tol=.2,maxdepth=10):
		self.data = data
		self.y = y
		self.depth = depth
		self.classes = classes
		self.tol = tol
		self.n_samples = float(len(y))
		self.variables = variables
		self.types = types
		self.n_variables = len(self.types)
		self.gini = gini(self.y,self.classes)
		self.maxdepth = maxdepth
		if self.gini > tol and self.depth != maxdepth:
			self.score,self.value,self.index,inds1,inds2,y1,y2 = self.optimalsplit()
			self.type = self.types[self.index]
			data1 = [data[i] for i in inds1]
			data2 = [data[i] for i in inds2]
			self.leftchild = Node(data1,y1,classes,variables,types,depth+1,tol,maxdepth)
			self.rightchild = Node(data2,y2,classes,variables,types,depth+1,tol,maxdepth)
			self.leaf = False
		else:
			self.labelprobs = sp.array([sum(y==classes[i]) for i in xrange(len(classes))])/self.n_samples
			self.label = classes[self.labelprobs.argmax()]
			self.leaf = True

	def numsplit(self,variable,value):
		inds1 = [i for i in xrange(int(self.n_samples)) if self.data[i][variable] <= value]
		inds2 = [i for i in xrange(int(self.n_samples)) if self.data[i][variable] > value]
		y1 = sp.copy(self.y[inds1])
		y2 = sp.copy(self.y[inds2])
		n1 = len(y1)
		n2 = len(y2)
		if (n1 == 0) or (n2 == 0):
			return 0, inds1, inds2, y1, y2
		return self.gini - (n1/self.n_samples)*gini(y1,self.classes) - (n2/self.n_samples)*gini(y2,self.classes), inds1, inds2, y1, y2

	def factorsplit(self,variable,factor):
		inds1 = [i for i in xrange(int(self.n_samples)) if self.data[i][variable] == factor]
		inds2 = [i for i in xrange(int(self.n_samples)) if self.data[i][variable] != factor]
		y1 = sp.copy(self.y[inds1])
		y2 = sp.copy(self.y[inds2])
		n1 = len(y1)
		n2 = len(y2)
		if (n1 == 0) or (n2 == 0):
			return 0, inds1, inds2, y1, y2
		return self.gini - (n1/self.n_samples)*gini(y1,self.classes) - (n2/self.n_samples)*gini(y2,self.classes), inds1, inds2, y1, y2

	def optimalnumsplit(self,variable):
		possibilities = sp.sort(sp.array(list(set([self.data[i][variable] for i in xrange(int(self.n_samples))]))))
		inds1 = [i for i in xrange(int(self.n_samples))]
		if len(possibilities) == 0:
			return 0, sp.array([0.]), inds1, [], sp.copy(self.y[inds1]),sp.array([])
		splitscore, inds1, inds2, y1, y2 = self.numsplit(variable,possibilities[0])
		splitscore = sp.array([splitscore])
		if len(possibilities) <= 1:
			return splitscore[0],possibilities[0],inds1,inds2,y1,y2
		else:
			value = sp.array([(possibilities[0] + possibilities[1])/2.])
			for i in range(1,len(possibilities)):
				tempsplitscore,tempinds1,tempinds2,tempy1,tempy2 = self.numsplit(variable,possibilities[i])
				if i == (len(possibilities)-1):
					tempvalue = possibilities[i]
				else:
					tempvalue = (possibilities[i] + possibilities[i+1])/2.
				if tempsplitscore > splitscore:
					splitscore = sp.copy([tempsplitscore])
					inds1 = sp.copy(tempinds1)
					inds2 = sp.copy(tempinds2)
					y1 = sp.copy(tempy1)
					y2 = sp.copy(tempy2)
					value = sp.copy([tempvalue])
			return splitscore[0], value[0], inds1, inds2, y1, y2

	def optimalfactorsplit(self,variable):
		possibilities = list(set([self.data[i][variable] for i in xrange(int(self.n_samples))]))
		splitscore, inds1, inds2, y1, y2 = self.factorsplit(variable,possibilities[0])
		splitscore = sp.array([splitscore])
		factor = sp.array([possibilities[0]])
		for i in range(1,len(possibilities)):
			tempsplitscore,tempinds1,tempinds2,tempy1,tempy2 = self.factorsplit(variable,possibilities[i])
			tempfactor = possibilities[i]
			if tempsplitscore > splitscore:
				splitscore = sp.copy([tempsplitscore])
				inds1 = sp.copy(tempinds1)
				inds2 = sp.copy(tempinds2)
				y1 = sp.copy(tempy1)
				y2 = sp.copy(tempy2)
				factor = sp.copy([tempfactor])
		return splitscore[0], factor[0], inds1, inds2, y1, y2

	def optimalsplit(self):
		if self.types[0] == "Factor":
			score,valfac,inds1,inds2,y1,y2 = self.optimalfactorsplit(0)
		else:
			score,valfac,inds1,inds2,y1,y2 = self.optimalnumsplit(0)
		index = 0
		for i in range(1,self.n_variables):
			if self.types[i] == "Factor":
				tempscore,tempvalfac,tempinds1,tempinds2,tempy1,tempy2 = self.optimalfactorsplit(i)
			else:
				tempscore,tempvalfac,tempinds1,tempinds2,tempy1,tempy2 = self.optimalnumsplit(i)
			if tempscore > score:
				score = sp.copy([tempscore])[0]
				inds1 = sp.copy(tempinds1)
				inds2 = sp.copy(tempinds2)
				y1 = sp.copy(tempy1)
				y2 = sp.copy(tempy2)
				valfac = sp.copy([tempvalfac])[0]
				index = i
		return score, valfac, index, inds1, inds2, y1, y2

	def printtree(self):
		if self.leaf:
			st = ""
			for i in xrange(len(self.classes)):
				st += self.classes[i] + ": {}".format(self.labelprobs[i]) + "\t"
			print "\t"*self.depth + st
		else:
			if self.types[self.index]=="Factor":
				print "\t"*self.depth + "If " + self.variables[self.index] + " = " + self.value + ":"
				self.leftchild.printtree()
				print "\t"*self.depth + "If " + self.variables[self.index] + " != " + self.value + ":"
				self.rightchild.printtree()
			else:
				print "\t"*self.depth + "If " + self.variables[self.index] + " <= " + str(self.value) + ":"
				self.leftchild.printtree()
				print "\t"*self.depth + "If " + self.variables[self.index] + " > " + str(self.value) + ":"
				self.rightchild.printtree()

	def predict(self,sample):
		if self.depth == self.maxdepth or self.gini < self.tol:
			return self.label
		elif self.type == "Factor":
			if sample[self.index]==self.value:
				return self.leftchild.predict(sample)
			else:
				return self.rightchild.predict(sample)
		else:
			if sample[self.index] <= self.value:
				return self.leftchild.predict(sample)
			else:
				return self.rightchild.predict(sample)

def misclassificationrate(predictions,truth):
	return sum(predictions!=truth)/float(len(truth))
