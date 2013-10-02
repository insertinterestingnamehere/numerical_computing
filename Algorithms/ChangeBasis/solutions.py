'''
    Invertible Affine Transformations
    For this lab, you need to turn in this solutions.py file. Also place the 
    horse.npy in your Google Drive folder for this section.
'''

import numpy as np
import math
from matplotlib import pyplot as plt

def plotNewOld(new, old):
    ''' This plotting script gives better results than the one provided in the lab
        text. Please use this to plot your figures.
    '''
    # Find the largest and smallest x- and y-values, used to set the size of the axes
    x_max = np.max([np.max(old[0]), np.max(new[0])])
    x_min = np.min([np.min(old[0]), np.min(new[0])])
    y_max = np.max([np.max(old[1]), np.max(new[1])])
    y_min = np.min([np.min(old[1]), np.min(new[1])])

    # Create the first subplot
    plt.subplot(2,1,1)
    plt.plot(old[0],old[1], linestyle='None', marker=',')
    plt.axis('equal')
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])

    # Create the second subplot
    plt.subplot(2,1,2)
    plt.plot(new[0], new[1],linestyle='None', marker=',')
    plt.axis('equal')
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])

    # Display result
    plt.show()

'''
    Problem 1
    Implement the function declaration below. The pts argument has shape (2,n),
    where the first row represents the x-coordinates and the second row the
    y-coordinates. The stretch argument is simply a one-dimensional numpy array
    with two entries--the first gives the scaling factor in the x-direction,
    and the second gives the scaling factor in the y-direction. You should
    implement the dilation as a matrix multiplication. Use np.diag to create
    an appropriate matrix from the stretch argument. Then use np.dot to perform
    the multiplication. Return the result.
'''
def dilation(pts, stretch):
    return np.dot(np.diag(stretch), pts)

'''
    Problem 2
    Implement the function declaration below. The pts argument is the same as above.
    The angle argument is a floating point number representing the angle of rotation
    in radians. Inside of the function you will need to create the appropriate
    rotation matrix, and then perform the requisite matrix multiplication, and
    return the result.
'''
def rotation(pts, angle):
    a = math.cos(angle)
    b = math.sin(angle)
    rot = np.array([a,-b,b,a]).reshape((2,2))
    return np.dot(rot, pts)

'''
    Problem 3
    Implement the function declaration below. You will need an if statement to deal with
    the possible values of the direction argument (0 means horizontal, 1 means vertical).
    The c argument is the amount of shearing.
'''
def shear(pts, c, direction):
    shear_mat = np.eye(2)
    if direction==0:
        shear_mat[0,1]=c
    else:
        shear_mat[1,0]=c
    return np.dot(shear_mat,pts)

'''
    Problem 4
    Implement the function declaration below. The axis argument is a one-dimensional
    array with two entries, which represents a vector pointing in the direction of
    the axis of rotation. You will need to use this array to create the appropriate
    reflection matrix, and then perform matrix multiplication.
'''
def reflection(pts, axis):
    a = ((axis**2)*np.array([1,-1])).sum()
    b = 2*axis.prod()
    c = (axis**2).sum()
    ref = (np.array([a,b,b,-a]).reshape((2,2)))/float(c)
    return np.dot(ref,pts)

'''
    Problem 5
    Implement the function declaration below. The shift argument is an array with two
    entries. To perform the translation, you simply need to add the shift array to
    the pts array, but before doing this, use the .reshape method on the shift array
    to ensure that it has the same shape as the example given in the text.
'''
def translation(pts, shift):
    return pts + shift.reshape((2,1))

'''
    Problem 6
    Implement the function declaration below. The t argument indicates time (in seconds),
    the w argument indicates angular velocity (in radians/seconds) and the v argument
    indicates directional speed (in meters/second). Hint: you will use a composition of
    rotation and translation in this problem. Make sure the function returns an array
    indicating the new position of p1 at time t.
'''
def trajectory(t,w,v):
    angle = t*w
    p = np.array([[0],[1]])
    shift = t*v*0.5*np.array([math.sqrt(2),math.sqrt(2)])
    return translation(rotation(p, angle), shift)

# Use this function to visualize the trajectory of the particle
def plotTrajectory():
    times = np.arange(0,10,.1)
    pos = np.zeros((2,len(times)))
    for i in range(len(times)):
        pos[:,i] = trajectory(times[i],np.pi,3).flatten()
    plt.scatter(pos[0], pos[1])
    plt.show()
    
# The following code will be used to test your solutions. This should be the only code
# in this script that actually runs. DO NOT ALTER! To make sure you pass-off, run this
# file from command line by typing:   python solutions.py
# You should see a series of plots pop up, reflecting the various affine transformations
# that you implemented.

x, y = np.load('horse.npy')
pts = np.array([x,y])

stretch = np.array([1.5,2])
plotNewOld(dilation(pts,stretch), pts)

angle = np.pi/3.0
plotNewOld(rotation(pts,angle), pts)

plotNewOld(shear(pts,0.8,1), pts)

axis = np.array([1,math.sqrt(3)])
plotNewOld(reflection(pts,axis), pts)

shift = np.array([1,-2])
plotNewOld(translation(pts, shift), pts)

plotTrajectory()
