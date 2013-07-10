import scipy as sp
import string
import os
import PCA

def loadStopwords(filename):
	""" This function loads the stopwords list from a .txt file. """
	""" This will be provided for the student. """
	infile = open(filename,'r')
	stopwords = infile.readlines()
	for i in xrange(len(stopwords)):
		stopwords[i] = stopwords[i].rstrip()
	return stopwords

def removeStopwords(terms,stopwords):
	""" This function removes from terms all occurrences of words in the list stopwords. """
	""" This will be provided for the student. """
	output = []
	for x in terms:
		if x not in stopwords:
			output.append(x)
	return output

def uniq(listinput):
	""" This finds the unique elements of the list listinput. """
	""" This will be provided for the student. """
	output = []
	for x in listinput:
		if x not in output:
			output.append(x)
	return output

def parseDocument(filename,stopwords=None):
	""" This strips all punctuation, line information, and makes everything lowercase in a .txt file,
	    separating words. If stopwords is a list of words, it will only return the words not in
	    the stopwords list. """
	""" This will be provided for the student. """
	infile = open(filename,'r')
	lines = infile.readlines()
	for i in xrange(len(lines)):
		lines[i] = lines[i].rstrip().translate(string.maketrans("",""),string.punctuation).lower()
	words = str.split(' '.join(lines))
	if stopwords != None:
		words = removeStopwords(words,stopwords)
	return words

def termlist(directory,stopwords=None):
	""" This function takes all .txt documents in the directory and returns the unique termlist. """
	""" This will be provided for the student. """
	filenames = os.listdir(directory)
	termlist = []
	for i in xrange(len(filenames)):
		termlist += parseDocument(directory + filenames[i],stopwords=stopwords)
	return uniq(termlist)

def termVector(filename,stopwords,termlist):
	""" The student must code this. """
	words = parseDocument(filename,stopwords=stopwords)
	wordVector = sp.zeros(len(termlist))
	for word in words:
		ind = termlist.index(word)
		wordVector[ind] += 1
	return wordVector

def termFrequencyMatrix(directory,stopwords,termlist):
	""" The student must code this. """
	filenames = sp.sort(os.listdir(directory))
	frequencyMatrix = sp.zeros((len(termlist),len(filenames)))
	for i in xrange(len(filenames)):
		frequencyMatrix[:,i] = termVector(directory + filenames[i],stopwords,termlist)
	return frequencyMatrix.astype(float)

def tfidf(termFrequency):
	""" The student must code this. """
	gf = sp.sum(termFrequency,axis=1).astype(float)
	p = (termFrequency.T/gf).T
	g = sp.sum(p*sp.log(p+1)/sp.log(len(p[0,:])),axis=1) + 1
	a = (sp.log(termFrequency + 1).T*g).T
	return a

def angleDistance(reducedMatrix,filenames,index=0):
	""" The student must code this. """
	T = len(reducedMatrix[:,0])
	dist = sp.zeros(T-1)
	doc_vec = reducedMatrix[index,:]
	for i in xrange(T):
		if i < index:
			dist[i] = 1 - sp.dot(doc_vec,reducedMatrix[i,:])/(sp.linalg.norm(doc_vec)*sp.linalg.norm(reducedMatrix[i,:]))
		if i > index:
			dist[i-1] = 1 - sp.dot(doc_vec,reducedMatrix[i,:])/(sp.linalg.norm(doc_vec)*sp.linalg.norm(reducedMatrix[i,:]))
	argmin = sp.argmin(dist)
	if argmin >= index:
		argmin += 1
	return filenames[argmin]

def reduceMatrix(A,percentage):
	eigen,V,SIGMA,X = PCA.PCA(A,percentage=percentage)
	return(sp.dot(V,SIGMA))
