import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import isingmodel
import matplotlib.pyplot as plt

def initialize():
    spinconfig = isingmodel.initialize(100)
    plt.imshow(spinconfig)
    plt.savefig('init.pdf')
    
initialize()
