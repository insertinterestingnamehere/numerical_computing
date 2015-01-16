from numpy import array, kron, eye, zeros, log, sqrt, inf, mean, std, allclose
from numpy.linalg import inv 
from scipy.stats.distributions import norm
from scipy.optimize import fmin
import numpy as np
# note that this implementation is zero indexed (first time is t=0), although
# the lab is not (first time is t=1) to conform in style with the Kalman
# filter lab

# datasets
a = """17.0 16.6 16.3 16.1 17.1 16.9 16.8 17.4 17.1 17.0 16.7 17.4 17.2 17.4 
 17.4 17.0 17.3 17.2 17.4 16.8 17.1 17.4 17.4 17.5 17.4 17.6 17.4 17.3
 17.0 17.8 17.5 18.1 17.5 17.4 17.4 17.1 17.6 17.7 17.4 17.8 17.6 17.5 
 16.5 17.8 17.3 17.3 17.1 17.4 16.9 17.3 17.6 16.9 16.7 16.8 16.8 17.2 
 16.8 17.6 17.2 16.6 17.1 16.9 16.6 18.0 17.2 17.3 17.0 16.9 17.3 16.8 
 17.3 17.4 17.7 16.8 16.9 17.0 16.9 17.0 16.6 16.7 16.8 16.7 16.4 16.5 
 16.4 16.6 16.5 16.7 16.4 16.4 16.2 16.4 16.3 16.4 17.0 16.9 17.1 17.1
 16.7 16.9 16.5 17.2 16.4 17.0 17.0 16.7 16.2 16.6 16.9 16.5 16.6 16.6 
 17.0 17.1 17.1 16.7 16.8 16.3 16.6 16.8 16.9 17.1 16.8 17.0 17.2 17.3
 17.2 17.3 17.2 17.2 17.5 16.9 16.9 16.9 17.0 16.5 16.7 16.8 16.7 16.7
 16.6 16.5 17.0 16.7 16.7 16.9 17.4 17.1 17.0 16.8 17.2 17.2 17.4 17.2
 16.9 16.8 17.0 17.4 17.2 17.2 17.1 17.1 17.1 17.4 17.2 16.9 16.9 17.0
 16.7 16.9 17.3 17.8 17.8 17.6 17.5 17.0 16.9 17.1 17.2 17.4 17.5 17.9
 17.0 17.0 17.0 17.2 17.3 17.4 17.4 17.0 18.0 18.2 17.6 17.8 17.7 17.2
 17.4"""
b = """-3 -5 7 3 -3 4 16 14 -3 2 6 1 -2 -1 -6 -1 -11 9 4 -4 -5 -3 -1 1 -2 2 -4 4 -3 0
2 1 -2 -1 -1 0 -2 1 0 0 -9 1 1 4 0 -4 6 8 7 2 -1 0 -4 6 1 2 5 -1 2 -3 -3 1 5 4
9 -2 3 -4 -1 6 4 4 -4 4 8 16 4 -4 -6 4 4 -4 4 -2 -4 -3 -1 -7 -15 10 13 2 -4 3 0
6 6 6 -2 0 3 11 0 -2 6 5 4 1 5 9 4 -4 -3 -11 2 -2 -4 2 9 0 4 0 -1 3 0 -3 0 -3
-4 -6 -6 2 11 -4 -5 -1 5 -3 0 -4 0 1 2 6 -3 -2 -5 -5 4 0 -2 4 5 2 -5 -7 5 -6
-11 -11 7 1 6 1 3 -6 -2 -6 0 0 -6 3 -6 -3 -9 -7 17 13 3 -7 0 1 1 4 0 -9 -1 -2 0
-6 0 -4 2 -2 1 1 6 5 -2 1 1 1 4 0 -1 -1 -1 3 1 -3 0 -6 2 0 -10 2 -1 -5 -8 -12
-3 11 0 0 2 -7 -5 7 -1 3 -1 0 0 -1 -5 -14 -14 -31 8 11 9 4 -11 -16 -8 2 -7 9 -3
5 -8 1 -15 -20 -17 1 -38 22 10 -8 -25 4 1 5 4 -15 -24 -12 -17 27 -3 6 -8 -12 4
12 -12 13 11 -5 11 1 -1 -5 5 9 16 4 -3 6 -12 -5 2 5 1 -10 8 -2 7 11 0 -11 9 0
-5 -7 9 -5 -1 3 7 1 -1 3 5 -1 16 2 -2 -1 -15 -2 -3 8 -9 -4 5 -6 2 -2 1 0 5 7 -3
-6 -3 -6 -13 5 -14 -5 3 -13 10 -1 9 2 0 6 -7 -3 -1 12 -10 4 -6 -7 -5 -13 10 -1
-8 14 7 -6 6 5"""
c = """ 101  82  66  35  31   7  20  92 154 125
  85  68  38  23  10  24  83 132 131 118
  90  67  60  47  41  21  16   6   4   7
  14  34  45  43  48  42  28  10   8   2
   0   1   5  12  14  35  46  41  30  24
  16   7   4   2   8  17  36  50  62  67
  71  48  28   8  13  57 122 138 103  86
  63  37  24  11  15  40  62  98 124  96
  66  64  54  39  21   7   4  23  55  94
  96  77  59  44  47  30  16   7  37  74"""
ta = np.fromstring(a, sep=' ',)
ta = ta.reshape((len(ta),1))
tb = np.fromstring(b, sep=' ')
tb = tb.reshape((len(tb),1))
tc = np.fromstring(c, sep=' ')
tc = tc.reshape((len(tc),1))

# fitted models
fit_a = (array([ 0.90867024]), array([-0.57585945]),
        17.065262486340927, 0.31253098628150655)
fit_b = (array([ 0.2350456, -0.3839864, -0.6566961]),
        array([-0.20234983,  0.41060419,  0.67314649]), -0.2853804404204241,
        7.0334525375368138)
fit_c = (array([ 1.22481184, -0.56007884]), array([ 0.38466735]),
        48.462278111207979, 14.622537558888457)

def arma_likelihood(time_series, phis=array([]), thetas=array([]), mu=0.,
        sigma=1.):
    """
    Return the log-likelihood of the ARMA model parameters, given the time
    series.

    Parameters
    ----------
    time_series : ndarray of shape (n,1)
        The time series in question
    phis : ndarray of shape (p,)
        The phi parameters
    thetas : ndarray of shape (q,)
        The theta parameters
    mu : float
        The parameter mu
    sigma : float
        The parameter sigma

    Returns
    -------
    log_likelihood : float
        The log-likelihood of the model
    """
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu,
            sigma)
    mus, covs = kalman(F, Q, H, time_series - mu)
    likelihood = 0.
    for i in xrange(len(mus)):
        cond_mu = H.dot(mus[i])
        cond_sigma = H.dot(covs[i].dot(H.T))
        likelihood += log(norm.pdf(time_series[i] - mu, loc=cond_mu,
            scale=sqrt(cond_sigma)))
    return float(likelihood)

def arma_fit(time_series):
    """
    Return the ARMA model that minimizes AICc for the given time series,
    subject to p,q <= 3. 

    Parameters
    ----------
    time_series : ndarray of shape (n,1)
        The time series in question

    Returns
    -------
    phis : ndarray of shape (p,)
        The phi parameters
    thetas : ndarray of shape (q,)
        The theta parameters
    mu : float
        The parameter mu
    sigma : float
        The parameter sigma
    """
    best_aicc = inf
    best_params = [], [], 0, 0
    emp_mean = mean(time_series)
    emp_sigma = std(time_series)
    for p in range(4):
        for q in range(4):
            print "Optimizing for p={}, q={}".format(p, q)
            x = array([0]*p + [0]*q + [emp_mean] + [emp_sigma])
            def f(x):
                return -1*arma_likelihood(time_series, phis=x[:p],
                        thetas=x[p:p+q], mu=x[-2], sigma=x[-1])
            opt_x = fmin(f, x, maxiter=10000, maxfun=10000)
            print "Optimal x {}".format(opt_x)
            aicc = 2*len(opt_x)*(1 + (len(opt_x) + 1)/(len(time_series) - \
                    len(opt_x))) + 2*f(opt_x)
            print "AICc {}".format(aicc)
            if aicc < best_aicc:
                print "New best model found with p={}, q={}".format(p, q)
                best_aicc = aicc
                best_params = opt_x[:p], opt_x[p:p+q], opt_x[-2], opt_x[-1]
    return best_params

def arma_forecast(time_series, phis=array([]), thetas=array([]), mu=0.,
        sigma=1., future_periods=20):
    """
    Return forecasts for a time series modeled with the given ARMA model.
    
    Parameters
    ----------
    time_series : ndarray of shape (n,1)
        The time series in question
    phis : ndarray of shape (p,)
        The phi parameters
    thetas : ndarray of shape (q,)
        The theta parameters
    mu : float
        The parameter mu
    sigma : float
        The parameter sigma
    future_periods : int
        The number of future periods to return

    Returns
    -------
    evls : ndarray of shape (future_periods,)
        The expected values of z for times n + 1, ..., n + future_periods
    sigs : ndarray of shape (future_periods,)
        The deviations of z for times n + 1, ..., n + future_periods
    """
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu,
            sigma)
    mus, covs = kalman(F, Q, H, time_series - mu)
    fut_covs = zeros((future_periods + 1, dim_states, dim_states))
    fut_mus = zeros((future_periods + 1, dim_states))
    evls = zeros(future_periods + 1)
    sigs = zeros(future_periods + 1)

    # forecast using Kalman filter
    yk = (time_series[-1] - mu) - H.dot(mus[-1])
    Sk = H.dot(covs[-1]).dot(H.T)
    Kk = covs[-1].dot(H.T.dot(inv(Sk)))
    fut_mus[0] = mus[-1] + Kk.dot(yk)
    fut_covs[0] = (eye(covs[-1].shape[0]) - Kk.dot(H)).dot(covs[-1])
    evls[0] = H.dot(fut_mus[0]) + mu
    sigs[0] = H.dot(fut_covs[0]).dot(H.T)
    for i in xrange(1, future_periods + 1):
        fut_mus[i] = F.dot(fut_mus[i-1])
        fut_covs[i] = F.dot(fut_covs[i-1]).dot(F.T) + Q
        evls[i] = H.dot(fut_mus[i]) + mu
        sigs[i] = sqrt(H.dot(fut_covs[i]).dot(H.T))
    return evls[1:], sigs[1:]

def kalman(F, Q, H, time_series):
    dim_time_series = time_series[0].shape[0]
    dim_states = F.shape[0]

    # covs[i] = P_{i | i-1}
    covs = zeros((len(time_series), dim_states, dim_states))
    mus = zeros((len(time_series), dim_states))

    covs[0] = inv(eye(dim_states**2) - kron(F,F)).dot(Q.flatten()).reshape(
            (dim_states,dim_states))
    mus[0] = zeros((dim_states,))

    for i in xrange(1, len(time_series)):
        t1 = inv(H.dot(covs[i-1]).dot(H.T))
        t2 = covs[i-1].dot(H.T.dot(t1.dot(H.dot(covs[i-1]))))
        covs[i] = F.dot((covs[i-1] - t2).dot(F.T)) + Q
        mus[i] = F.dot(mus[i-1]) + F.dot(covs[i-1].dot(H.T.dot(t1))).dot(
                time_series[i-1] - H.dot(mus[i-1]))
    return mus, covs

def state_space_rep(phis, thetas, mu, sigma):
    dim_states = max(len(phis), len(thetas)+1)
    dim_time_series = 1 #hardcoded for 1d time_series

    F = zeros((dim_states,dim_states))
    Q = zeros((dim_states, dim_states))
    H = zeros((dim_time_series, dim_states))

    F[0][:len(phis)] = phis
    F[1:,:-1] = eye(dim_states - 1)
    Q[0][0] = sigma**2
    H[0][0] = 1.
    H[0][1:len(thetas)+1] = thetas

    return F, Q, H, dim_states, dim_time_series

def test(time_series_a):
    """Assert that statements made in lab examples are correct"""
    assert allclose(arma_likelihood(time_series_a, array([0.9]), array([]),
        17., 0.4), -77.603545, atol=1e-4)
    phis, thetas, mu, sigma = fit_time_series_a
    evls, sigs = arma_forecast(time_series_a, phis, thetas, mu, sigma,
            future_periods=4)
    assert allclose(evls, (17.3762, 17.3478, 17.322, 17.2986), atol=1e-4)
    assert allclose(sigs, (0.3125, 0.3294, 0.3427, 0.3533), atol=1e-4)
    fit_test = arma_fit(time_series_a)
    for i in xrange(4):
        assert allclose(fit_test[i], fit_time_series_a[i], atol=1e-4)
