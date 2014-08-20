#Problem 1
import scipy as sp
from scipy import stats as st

def discretenorm(K,mu,sigma):
    upper = mu + 3*sigma
    lower = mu - 3*sigma
    
    inc = (upper - lower)/float(K)

    left = lower    
    prob = sp.zeros(K)
    eps  = sp.zeros(K)
    for k in range(K):
        prob[k] = st.norm.cdf(left+inc,mu,sigma) - st.norm.cdf(left,mu,sigma)
        eps[k] = left + .5*inc
        left = left + inc
        
    return prob, eps