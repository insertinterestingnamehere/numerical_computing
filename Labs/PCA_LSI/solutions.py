import numpy as np
from collections import Counter
from math import log
import string
from scipy import sparse
from scipy.sparse import linalg as sparla
from matplotlib import pyplot as plt
from numpy.linalg import norm
from itertools import izip
from scipy import linalg as la
import sklearn.datasets as datasets

def PCA_Code():
    # load iris data
    iris = datasets.load_iris()
    X = iris.data
    # pre-process
    Y = X - X.mean(axis=0)
    # get SVD
    U,S,VT = la.svd(Y,full_matrices=False)
    # project onto the first two principal components
    Yhat = U[:,:2].dot(np.diag(S[:2]))
    # plot results
    setosa = iris.target==0
    versicolor = iris.target==1
    virginica = iris.target==2
    p1, p2 = Yhat[:,0], Yhat[:,1]
    plt.scatter(p1[setosa],p2[setosa], marker='.', color='blue', label='Setosa')
    plt.scatter(p1[versicolor],p2[versicolor], marker='.', color='red', label='Versicolor')
    plt.scatter(p1[virginica],p2[virginica], marker='.', color='green', label='Virginica')
    plt.legend(loc=2)
    plt.ylim([-4,5])
    plt.xlim([-4,4])
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

def LSI_Code():
    # create vocab for the state of the union addresses
    from os import listdir
    # get list of filepaths to each text file in the folder
    path_to_addresses = "./Addresses/"
    paths = [path_to_addresses + p for p in os.listdir(path_to_addresses) if p[-4:]==".txt"]
    
    def extractWords(text):
        return text.strip().translate(trans, string.punctuation+string.digits).lower().split()
    
    # initialize vocab set, then read each file and add to the vocab set
    vocab = set()
    trans = string.maketrans("", "")
    for p in paths:
        with open(p, 'r') as f:
            for line in f:
                vocab.update(extractWords(line))
    
    # load stopwords
    with open("stopwords.txt", 'r') as f:
        stopwords = set([w.strip().lower() for w in f.readlines()])
    # remove stopwords from vocabulary
    vocab = {w:i for i, w in enumerate(vocab.difference(stopwords))}
    
    speech_indices = [50,30] # indices relative to paths of speeches to query
    
    # basic term frequency approach
    counts = []
    doc_index = []
    word_index = []
    
    for doc, p in enumerate(paths):
        with open(p, 'r') as f:
            # create the word counter
            ctr = Counter()
            for line in f:
                words = extractWords(line)
                ctr.update(words)
                
            # iterate through the word counter, store counts
            for word, count in ctr.iteritems(): 
                try: # only look at words in vocab
                    word_index.append(vocab[word])
                    counts.append(count)
                    doc_index.append(doc)
                except KeyError:
                    pass
            
    # create sparse matrix holding these word counts
    X = sparse.csr_matrix((counts, [doc_index,word_index]), shape=(len(paths),len(vocab)), dtype=np.float)
    
    # perform LSI on X
    U,S,VT = sparla.svds(X, k=7)
    # project onto principle components space
    Xhat = U.dot(np.diag(S))
    # normalize the rows
    Xhat /= norm(Xhat, axis=1)[:,np.newaxis]
    # get inner products
    angles = Xhat.dot(Xhat[speech_indices,:].T)
    # find closest speeches
    for i,ind in enumerate(speech_indices):
        print "Closest speech to {} is {}".format(paths[ind][12:-4], paths[angles[:,i].argsort()[-2]][12:-4])
        print "Furthest speech to {} is {}".format(paths[ind][12:-4], paths[angles[:,i].argsort()[0]][12:-4])
        
    # tfidf approach
    total_freqs = np.zeros(len(vocab))
    counts = []
    doc_index = []
    word_index = []
    
    # get doc-term freqs and global term freqs
    for doc, p in enumerate(paths):
        with open(p, 'r') as f:
            # create the word counter
            ctr = Counter()
            for line in f:
                words = extractWords(line)
                ctr.update(words)
            # iterate through the word counter, store counts
            for word, count in ctr.iteritems(): 
                try: # only look at words in vocab
                    word_ind = vocab[word]
                    word_index.append(word_ind)
                    counts.append(count)
                    doc_index.append(doc)
                    total_freqs[word_ind] += count
                except KeyError:
                    pass
                
    # get global weights
    global_weights = np.ones(len(vocab))
    logM = log(len(paths))
    for count, word in izip(counts, word_index):
        p = count/float(total_freqs[word])
        global_weights[word] += p*log(p+1)/logM
    
    # get globally weighted counts
    gwcounts = []
    for count, word in izip(counts, word_index):
        gwcounts.append(global_weights[word]*log(count+1))
                    
    # create sparse matrix holding these globally weighted word counts
    X = sparse.csr_matrix((gwcounts, [doc_index,word_index]), shape=(len(paths),len(vocab)), dtype=np.float)
    
    # perform LSI on X
    U,S,VT = sparla.svds(X, k=7)
    # project onto principle components space
    Xhat = U.dot(np.diag(S))
    # normalize the rows
    Xhat /= norm(Xhat, axis=1)[:,np.newaxis]
    # get inner products
    angles = Xhat.dot(Xhat[speech_indices,:].T)
    # find closest speeches
    for i,ind in enumerate(speech_indices):
        print "Closest speech to {} is {}".format(paths[ind][12:-4], paths[angles[:,i].argsort()[-2]][12:-4])
        print "Furthest speech to {} is {}".format(paths[ind][12:-4], paths[angles[:,i].argsort()[0]][12:-4])

def LSI_answers():
    print """For basic term frequency matrix, we have:
    Closest speech to 1993-Clinton is 2010-Obama
    Furthest speech to 1993-Clinton is 1951-Truman
    Closest speech to 1974-Nixon is 1988-Reagan
    Furthest speech to 1974-Nixon is 2001-GWBush-1
    
    For tfidf matrix, we have:
    Closest speech to 1993-Clinton is 1992-Bush
    Furthest speech to 1993-Clinton is 1946-Truman
    Closest speech to 1974-Nixon is 1972-Nixon
    Furthest speech to 1974-Nixon is 1946-Truman"""
    
