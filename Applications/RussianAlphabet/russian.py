import unicodedata
import codecs
import sys
import scipy as sp
import hmm

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))

def uniq(listinput):
	""" This finds the unique elements of the list listinput. """
	""" This will be provided for the student. """
	output = []
	for x in listinput:
		if x not in output:
			output.append(x)
	return output

def remove_punctuation(text):
    return text.translate(tbl)

def fileParse(filename):
	infile = codecs.open(filename,'r','utf-8')
	data = []
	for line in infile:
		data.append(line)
	characters = []
	for i in xrange(len(data)):
		characters += list(remove_punctuation(data[i].rstrip() + ' ').lower())
	return(characters)

def alphabet(characters):
	return(sp.sort(uniq(characters)).tolist())

def printLetters(alphabet):
	for i in xrange(len(alphabet)):
		print alphabet[i].encode('utf-8')
	return

def russianHMM(n_trials,characters,alphabet):
	model = hmm.hmmTrain(characters,2,tol=.1,maxiter=100)
	obs = hmm.transformObservations(alphabet,characters)
	logProb = hmm.score(model,obs)
	for i in xrange(n_trials-1):
		print(i)
		temp_model = hmm.hmmTrain(characters,2,tol=.1,maxiter=100)
		temp_logProb = hmm.score(temp_model,obs)
		if temp_logProb > logProb:
			model = temp_model
			logProb = temp_logProb
	return model

def separateVowels(model,characters,alphabet):
	alpha,beta,logProb = hmm.betaPass(model,hmm.transformObservations(alphabet,characters))
	gam = hmm.gamma(alpha,beta)
	stateProbs = sp.sum(gam,0)/gam.shape[0]
	bools = model[1][0,:]*stateProbs[0] > model[1][1,:]*stateProbs[1]
	state_1_chars = sp.array(alphabet)[bools].tolist()
	state_2_chars = sp.array(alphabet)[-bools].tolist()
	return state_1_chars, state_2_chars
