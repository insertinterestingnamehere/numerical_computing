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

class Forest(object):

	def __init__(self,data,y,classes,variables,types,maxdepth,tol):
		self.data = data
		self.y = y
		self.classes = classes
		self.n_samples = float(len(y))
		self.variables = variables
		self.types = types
		self.n_variables = len(types)
		self.maxdepth = maxdepth
		self.tol = tol
		self.availablevars = sp.array([True for i in xrange(self.n_variables)])

	def train(self,mtry,n_trees):
		self.mtry = mtry
		self.n_trees = n_trees
		self.trees = []
		for i in xrange(n_trees):
			print i
			self.trees.append(ForestNode(self.data,self.y,self.classes,self.variables,self.availablevars,self.types,mtry,self.tol,1,self.maxdepth))

	def predict(self,sample):
		predictions = [self.trees[i].predict(sample) for i in xrange(self.n_trees)]
		return stats.mode(predictions)[0][0]

class ForestNode(object):

	def __init__(self,data,y,classes,variables,available_vars,types,mtry,tol=.1,depth=1,maxdepth=10):
		self.data = data
		self.mtry = mtry
		self.y = y
		self.tol = tol
		self.classes = classes
		self.n_samples = float(len(y))
		self.variables = variables
		self.types = types
		self.n_variables = len(self.types)
		self.available_vars = available_vars
		self.n_available_vars = sum(available_vars)
		self.depth = depth
		self.maxdepth = maxdepth
		self.gini = gini(self.y,self.classes)
		if self.depth < self.maxdepth and self.gini > self.tol and self.n_available_vars >= mtry:
			self.score,self.value,self.index,inds1,inds2,y1,y2 = self.optimalsplit()
			self.type = self.types[self.index]
			data1 = [data[i] for i in inds1]
			data2 = [data[i] for i in inds2]
			n1 = len(set([x[self.index] for x in data1]))
			n2 = len(set([x[self.index] for x in data2]))
			tempavailablevars = sp.copy(self.available_vars)
			for i in xrange(self.n_variables):
				if self.available_vars[i]:
					if len(set([x[i] for x in data1])) == 1:
						tempavailablevars[i] = False
			self.leftchild = ForestNode(data1,y1,classes,variables,tempavailablevars,types,mtry,tol,depth+1,maxdepth)
			tempavailablevars = sp.copy(self.available_vars)
			for i in xrange(self.n_variables):
				if self.available_vars[i]:
					if len(set([x[i] for x in data2])) == 1:
						tempavailablevars[i] = False
			self.rightchild = ForestNode(data2,y2,classes,variables,tempavailablevars,types,mtry,tol,depth+1,maxdepth)
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
		if len(possibilities) <= 1:
			return splitscore, possibilities[0], inds1, inds2, y1, y2
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
		inds = [i for i in xrange(self.n_variables) if self.available_vars[i]]
		vars = sp.sort(random.sample(inds,self.mtry))
		if self.types[vars[0]] == "Factor":
			score,valfac,inds1,inds2,y1,y2 = self.optimalfactorsplit(vars[0])
		else:
			score,valfac,inds1,inds2,y1,y2 = self.optimalnumsplit(vars[0])
		index = vars[0]
		for i in vars[1:]:
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
		if self.depth == self.maxdepth or self.gini < self.tol or self.n_available_vars < self.mtry:
			st = ""
			for i in xrange(len(self.classes)):
				st += self.classes[i] + ": {}".format(self.labelprobs[i]) + "\t"
			print "\t"*self.depth + st + "\tGini Impurity: {}".format(self.gini)
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
		if self.depth == self.maxdepth or self.gini < self.tol or self.n_available_vars < self.mtry:
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
