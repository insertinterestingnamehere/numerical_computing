import scipy as sp
import scipy.io.wavfile as wavfile
import os
import gmmhmm as hmm
import MFCC
import numpy as np

def sample_gmmhmm(gmmhmm, n_sim):
    """
    Simulate from a GMMHMM.
    
    Returns
    -------
    states : ndarray of shape (n_sim,)
        The sequence of states
    obs : ndarray of shape (n_sim, K)
        The generated observations (vectors of length K)
    """
    states = []
    obs = []
    state = np.argmax(np.random.multinomial(1,gmmhmm[-1]))
    for i in xrange(n_sim):
        sample_component = np.argmax(np.random.multinomial(1, gmmhmm[1][state,:]))
        sample = np.random.multivariate_normal(gmmhmm[2][state, sample_component, :], gmmhmm[3][state, sample_component, :, :])
        states.append(state)
        obs.append(sample)
        state = np.argmax(np.random.multinomial(1, gmmhmm[0][:,state]))
    return np.array(states), np.array(obs)

def collect(n=20):
	obs = []
	for i in xrange(n):
		os.system("arecord -f S16_LE --rate=44100 -D hw:1,0 -d 3 test.wav")
		obs.append(MFCC.extract(wavfile.read("test.wav")[1]))
	return obs

def initialize(n_states):
	transmat = sp.ones((n_states,n_states))/float(n_states)
	for i in xrange(n_states):
		transmat[i,:] += sp.random.uniform(-1./n_states,1./n_states,n_states)
		transmat[i,:] /= sum(transmat[i,:])
	startprob = sp.ones(n_states)/float(n_states) + sp.random.uniform(-1./n_states,1./n_states,n_states)
	startprob /= sum(startprob)
	return transmat,startprob

def modelTrain(n_states,n_mix,obs,n_trials=10):
	transmat,startprob = initialize(n_states)
	model = hmm.GMMHMM(n_components=n_states,n_mix=n_mix,transmat=transmat,startprob=startprob,cvtype='diag')
	model.covars_prior = 0.01
	model.fit(obs,init_params='mc',var=.1)
	like = model.logprob
	for i in sp.arange(1,n_trials):
		print i
		transmat,startprob = initialize(n_states)
		temp_model = hmm.GMMHMM(n_states,n_mix,transmat=transmat,startprob=startprob,cvtype='diag')
		temp_model.covars_prior = 0.1
		temp_model.fit(obs,init_params='mc',var=.1)
		if temp_model.logprob > like:
			model = temp_model
			like = temp_model.logprob
	return model

def detect(models,model_names,filename):
	os.system("arecord -f S16_LE --rate=44100 -D hw:1,0 -d 2 " + filename)
	obs = wavfile.read(filename)[1]
	obs = processWav(obs)
	n_models = len(models)
	scores = sp.zeros(n_models)
	for i in xrange(n_models):
		scores[i] = models[i].score(obs)
	n = sp.argmax(scores)
	print "Word: " + model_names[n]
	return
