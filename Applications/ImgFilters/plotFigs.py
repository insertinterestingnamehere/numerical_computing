import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from matplotlib import pyplot as plt

K = plt.imread('cameraman.tif')
blur = np.array([[2,4,5,4,2],
                 [4,9,12,9,4],
                 [5,12,15,12,5],
                 [4,9,12,9,4],
                 [2,4,5,4,2]])/159.
                 
def Filter(I, f):
    '''
    Filter an image I with matrix filter f.
    Inputs:
        f -- numpy array of shape (m,n), where m and n are odd
        I -- numpy array of shape (k,l), the image
    Outpus:
        O -- numpy array of shape (k,l), the filtered image
    '''
    m,n = f.shape
    k,l = I.shape
    
    #initialize the output array
    O = np.empty((k,l))
    
    #create a padded version of I by padding with zeros
    N = np.zeros((k+m-1, l+n-1))
    N[m/2:m/2+k, n/2:n/2+l] = I

    #set the ouput pixels
    for i in xrange(k):
        for j in xrange(l):
            O[i,j] = (f*N[i:i+m,j:j+n]).sum()
    return O

def plotEdges(I):
    '''
    Find and plot the edges of an image I using the Sobel filter.
    Inputs:
        I -- numpy array of shape (m,n), the image
    Returns:
        This function returns nothing, but the last line is a call
        to the function plt.show() from matplotlib.pyplot
    '''
    #create the Sobel filter (for the vertical gradient)
    S = np.array([[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]])/8.
    
    #filter the image horizontally and vertically to get gradient values
    Oy = Filter(K, Sobel)
    Ox = Filter(K, Sobel.T)
    
    #combine to obtain gradient magnitude at each pixel
    O = np.sqrt(Oy**2+Ox**2)
    
    #set threshold value
    thresh = 4*O.mean()
    
    #plot the thresholded image
    plt.imshow(O > thresh, cmap=plt.cm.Greys_r)
    plt.show()
    
def cameramanClean():
    plt.imshow(K, cmap=plt.cm.Greys_r)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=False)
    plt.savefig('cameramanClean.pdf')
    plt.clf()
    
def cameramanBlur():
    O = Filter(K, blur)
    plt.imshow(O, cmap=plt.cm.Greys_r)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=False)
    plt.savefig('cameramanBlur.pdf')
    plt.clf()

def edges():
    S = np.array([[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]])/8.
    
    #filter the image horizontally and vertically to get gradient values
    Oy = Filter(K, S)
    Ox = Filter(K, S.T)
    
    #combine to obtain gradient magnitude at each pixel
    O = np.sqrt(Oy**2+Ox**2)
    
    #set threshold value
    thresh = 4*O.mean()
    
    #plot the thresholded image
    plt.imshow(O > thresh, cmap=plt.cm.Greys_r)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=False)
    plt.savefig('edges.pdf')
    plt.clf()

cameramanClean()
cameramanBlur()
edges()
