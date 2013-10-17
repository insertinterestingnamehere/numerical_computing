import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
from matplotlib import pyplot as plt

import numpy as np
import solutions.py 

x = [-1.5, -1., -.5, 0., .5, 1., 1.5, .75, -.75]
y = [0., -1., -2., -2., -2., -1., 0., 2., 2.]

A = np.array([x, y])

def plotFigs(oldA, newA, figName):
    plt.subplot(2, 1, 1)
    plt.scatter(oldA[0], oldA[1])
    plt.axis('equal')
    plt.subplot(2, 1, 2)
    plt.scatter(newA[0], newA[1])
    plt.axis('equal')
    plt.savefig(figName)
    plt.clf()

def stretch():
    plotFigs(A, basis.strc(A, np.array([2,1.5])), 'stretch.pdf')

def rotate():
    plotFigs(A, basis.rotate(A, 3*np.pi/16.0), 'rotate.pdf')
    
def shear():
    plotFigs(A, basis.shear(A, 0, .5), 'shear.pdf')
    
def reflect():
    plotFigs(A, basis.reflect(A, np.array([-1,.3])), 'reflection.pdf')
    
def translate():
    plotFigs(A, basis.shift(A, np.array([4, -2])), 'shift.pdf')
    
def combo():
    plotFigs(A, basis.combo(A, 2, .5, np.pi/3, -3, 1), 'combo.pdf')

stretch()
rotate()
shear()
reflect()
translate()
combo()
