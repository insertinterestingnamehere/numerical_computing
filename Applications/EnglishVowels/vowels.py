import scipy as sp
import re
import string
import pickle
import hmmstates

def parseDocument(filename):
	""" This strips all punctuation, line information, and makes everything lowercase in a .txt file,
	    separating words. If stopwords is a list of words, it will only return the words not in
	    the stopwords list. """
	""" This will be provided for the student. """
	infile = open(filename,'r')
	lines = infile.readlines()
	for i in xrange(len(lines)):
		lines[i] = re.sub('[0-9]+','',lines[i].rstrip().translate(string.maketrans("",""),string.punctuation).lower())
	words = ' '.join(lines)
	return list(words)

def vowelConsonantClassifier(characters):
	vow = ['a','e','i','o','u',' ']
	T = len(characters)
	trueStates = sp.ones(T)
	for i in xrange(T):
		if characters[i] in vow:
			trueStates[i] = 0
	return(trueStates)

def misclassificationRate(trueStates,stateEstimates):
	T = len(trueStates)
	return(sum(trueStates - stateEstimates != 0)/float(T))

infile = open('englishHMM','r')
englishHMM = pickle.load(infile)
alphabet = englishHMM[3]
declaration = parseDocument('declaration.txt')
stateEstimates = hmmstates.stateEstimation(englishHMM,hmmstates.transformObservations(alphabet,declaration))
trueStates = vowelConsonantClassifier(declaration)
misclassRate = misclassificationRate(trueStates,stateEstimates)
