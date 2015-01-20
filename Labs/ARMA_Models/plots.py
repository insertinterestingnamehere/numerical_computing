import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt
import arma_solution as sol

def forecastPlots():
    s1 = sol.arma_forecast(sol.ta,phis=sol.fit_a[0], thetas=sol.fit_a[1],mu=sol.fit_a[2], sigma=sol.fit_a[3], future_periods=20)
    s2 = sol.arma_forecast(sol.tb,phis=sol.fit_b[0], thetas=sol.fit_b[1],mu=sol.fit_b[2], sigma=sol.fit_b[3], future_periods=20)
    s3 = sol.arma_forecast(sol.tc,phis=sol.fit_c[0], thetas=sol.fit_c[1],mu=sol.fit_c[2], sigma=sol.fit_c[3], future_periods=20)
    l = len(sol.ta)
    plt.plot(np.arange(l), sol.ta, 'b')
    plt.plot(np.arange(l, l+20), s1[0], 'g')
    plt.plot(np.arange(l, l+20), s1[0]+s1[1], 'y--')
    plt.plot(np.arange(l, l+20), s1[0]-s1[1], 'y--')
    plt.plot(np.arange(l, l+20), s1[0]+2*s1[1], 'c--')
    plt.plot(np.arange(l, l+20), s1[0]-2*s1[1], 'c--')
    plt.savefig("forecasted_a.pdf")
    plt.clf()
    
    l = len(sol.tb)
    plt.plot(np.arange(l), sol.tb, 'b')
    plt.plot(np.arange(l, l+20), s2[0], 'g')
    plt.plot(np.arange(l, l+20), s2[0]+s2[1], 'y--')
    plt.plot(np.arange(l, l+20), s2[0]-s2[1], 'y--')
    plt.plot(np.arange(l, l+20), s2[0]+2*s2[1], 'c--')
    plt.plot(np.arange(l, l+20), s2[0]-2*s2[1], 'c--')
    plt.savefig("forecasted_b.pdf")
    plt.clf()
    
    l = len(sol.tc)
    plt.plot(np.arange(l), sol.tc, 'b')
    plt.plot(np.arange(l, l+20), s3[0], 'g')
    plt.plot(np.arange(l, l+20), s3[0]+s3[1], 'y--')
    plt.plot(np.arange(l, l+20), s3[0]-s3[1], 'y--')
    plt.plot(np.arange(l, l+20), s3[0]+2*s3[1], 'c--')
    plt.plot(np.arange(l, l+20), s3[0]-2*s3[1], 'c--')
    plt.savefig("forecasted_c.pdf")
    plt.clf()
if __name__ == "__main__":
    forecastPlots()
