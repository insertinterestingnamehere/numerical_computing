def sim_data(x, k):
    """
    Generate samples from beta distributions whose parameters are given
    by the rows of x.
    Inputs:
        x -- nx2 array
        k -- number of samples to generate for each row of x
    Outpus:
        samples -- kxn array, each column consists of k independent
                   draws from a beta distribution with parameters given
                   by the corresponding row of x.
    """
    n = x.shape[0]
    samples = np.empty((k,n))
    for i in xrange(n):
        samples[:,i] = np.random.beta(x[i,0], x[i,1], k)
    return samples

def arm_probs(x):
    """
    Calculate probabilities of each arm being the best.
    Inputs:
        x -- k by n array, where the i-th column gives
             k draws for p_i
    Returns:
        probs -- length n array. The i-th entry is the number
                 of rows of x in which the draw for p_i is 
                 the largest in its row, divided by k.
    """
    k, n = x.shape
    amx = np.argmax(x, axis=1)
    probs = np.zeros(n)
    for i in xrange(n):
        probs[i] = (amx==i).sum()
    return probs/k
    
def get_pulls(x, M):
    """
    Compute the number of times to pull each arm
    in the next M pulls, based on the probability
    that each arm is the best.
    Inputs:
        x -- length n stochastic array giving probabilities
             of each of the n arms being the best.
        M -- number of pulls to make
    Returns:
        pulls -- length n array giving the number of pulls for 
                 each arm. Entries must sum to M.
    """
    n = x.size
    pulls = np.floor(M*np.ones(n)*x)
    excess = M - pulls.sum()
    ind = np.random.permutation(n)[:excess]
    pulls[ind] += 1
    
    return pulls
    
def sim_day(cvr, state):
    """
    Simulate one day in the web page experiment.
    Inputs:
        cvr -- length n flat array, giving true CvR for each webpage
        state -- n x 2 array giving the beta parameters for each webpage
    Returns:
        state -- n x 2 array giving updated beta parameters
        pulls -- length n flat array giving number of pulls assigned to each arm
                 in the final 50 pulls of the day.
    """
    state = state.copy()
    n = state.shape[0]
    for j in xrange(2):
        x = sim_data(state, 100)
        weights = arm_probs(x)
        pulls = get_pulls(weights, 50)
        for i in xrange(n):
            conversions = np.random.binomial(pulls[i], cvr[i]) 
            state[i, 0] += conversions
            state[i,1] += pulls[i] - conversions
    return state, pulls
    
    
def val_remaining(data, prob):
    """
    Calculate the potential remaining value from simulated data and
    probility vector.
    """
    champ_ind = np.argmax(prob)
    thetaM = np.amax(data,1)
    valrem = (thetaM - data[:,champ_ind])/data[:,champ_ind]
    pvr = sp.stats.mstats.mquantiles(valrem ,.95)
    return pvr
    
def sim_experiment(cvr):
    """
    Simulate the webpage design experiment based on the Google Analytics approach.
    Inputs:
        cvr -- length n flat array of true CvR for each webpage.
    Returns:
        state -- n x 2 array giving final beta parameters for each webpage.
        pulls -- d x n array (d is the number of iterations) giving the 
                 pulls assigned to each webpage in each iteration.
        best -- integer, giving the index of the best webpage.
        days -- number of iterations.
    """
    n_sim = 100
    n = cvr.size
    state = np.ones((n,2))
    days = 0
    max_prob = 0
    max_pbar = 0
    pvr = 1
    pull_list = []
    while (days<14) or (max_prob <= .95 and max_pbar/100. <= pvr):
        days += 1
        state, pulls = sim_day(cvr, state)
        pull_list.append(pulls)
        data = sim_data(state, n_sim)
        probs = arm_probs(data)
        max_ind = np.argmax(probs)
        max_prob = probs[max_ind]
        max_pbar = state[max_ind, 0]/(state[max_ind,:].sum())
        pvr = val_remaining(data, probs)
    return state, np.array(pull_list),max_ind, 

#the code for plotting the results is already included in the lab.
