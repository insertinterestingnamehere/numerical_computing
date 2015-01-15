import numpy as np
import scipy as sp
import string
import os
import PCA

def loadStopwords(filename):
    """ This function loads the stopwords list from a .txt file.
    This will be provided for the student. """
    with open(filename, 'r') as infile:
        stopwords = set(x.rstrip() for x in infile.readlines())
    return stopwords

def removeStopwords(terms, stopwords):
    """ This function removes from terms all occurrences of words in the list stopwords.
    This will be provided for the student. """
    return [x for x in terms if x not in stopwords]

def uniq(listinput):
    """ This finds the unique elements of the list listinput.
    This will be provided for the student. """
    return list(set(listinput))

def parseDocument(filename, stopwords=[]):
    """ This strips all punctuation, line information, and makes everything lowercase in a .txt file,
        separating words. If stopwords is a list of words, it will only return the words not in
        the stopwords list.
    This will be provided for the student. """
    words = []
    trans = string.maketrans("", "")
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.rstrip().translate(trans, string.punctuation).lower()
            words.extend(removeStopwords(line.split(), stopwords))
    return words

def termlist(directory,stopwords=None):
    """ This function takes all .txt documents in the directory and returns the unique termlist.
    This will be provided for the student. """
    filenames = os.listdir(directory)
    termlist = []
    for i in xrange(len(filenames)):
        termlist += parseDocument(directory + filenames[i],stopwords=stopwords)
    return uniq(termlist)

def termVector(filename, stopwords, termlist):
    """ The student must code this. """
    words = parseDocument(filename, stopwords=stopwords)
    wordVector = np.zeros(len(termlist))
    for word in words:
        ind = termlist.index(word)
        wordVector[ind] += 1
    return wordVector

def termFrequencyMatrix(directory,stopwords,termlist):
    """ The student must code this. """
    filenames = np.sort(os.listdir(directory))
    frequencyMatrix = np.zeros((len(termlist),len(filenames)))
    for i in xrange(len(filenames)):
        frequencyMatrix[:,i] = termVector(directory + filenames[i],stopwords,termlist)
    return frequencyMatrix.astype(float)

def tfidf(termFrequency):
    """ The student must code this. """
    gf = np.sum(termFrequency,axis=1).astype(float)
    p = (termFrequency.T/gf).T
    g = np.sum(p*np.log(p+1)/np.log(len(p[0,:])),axis=1) + 1
    a = (np.log(termFrequency + 1).T*g).T
    return a

def angleDistance(reducedMatrix,filenames,index=0):
    """ The student must code this. """
    T = len(reducedMatrix[:,0])
    dist = np.zeros(T-1)
    doc_vec = reducedMatrix[index,:]
    for i in xrange(T):
        if i < index:
            dist[i] = 1 - np.dot(doc_vec,reducedMatrix[i,:])/(np.linalg.norm(doc_vec)*np.linalg.norm(reducedMatrix[i,:]))
        if i > index:
            dist[i-1] = 1 - np.dot(doc_vec,reducedMatrix[i,:])/(np.linalg.norm(doc_vec)*np.linalg.norm(reducedMatrix[i,:]))
    argmin = np.argmin(dist)
    if argmin >= index:
            argmin += 1
    return filenames[argmin]

def reduceMatrix(A,percentage):
    eigen,V,SIGMA,X = PCA.PCA(A,percentage=percentage)
    return(np.dot(V,SIGMA))
