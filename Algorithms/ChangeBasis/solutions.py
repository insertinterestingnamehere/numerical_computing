'''
    Invertible Affine Transformations
    For this lab, you need to turn in this solutions.py file. Also place the 
    horse.npy in your Google Drive folder for this section.
'''

import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import linalg as la

def plotNewOld(new, old):
    ''' 
    This plotting script gives better results than the one provided in the lab
    text. Please use this to plot your figures.
    Inputs:
    new -- a (2,n) numpy array containing the transformed x-coordinates on the 
            first row, y-coordinates on the second row.
    old -- a (2,n) numpy array containing the original x-coordinates on the first
            row, y-coordinates on the second row.
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


def dilate(pts, stretch):
    '''
    Apply a dilation to an array of points. You should
    implement the dilation as a matrix multiplication. Use np.diag to create
    an appropriate matrix from the stretch argument.
    Inputs:
        pts -- a (2,n) array of points, x-coordinates on the first row, y-coordinates
                on the second row.
        stretch -- a length 2 NumPy array giving the dilation factor in the x- and
                    y-direction entrywise, respectively
    Returns:
        a (2,n) array representing the dilated points.
    '''
    return np.dot(np.diag(stretch), pts)


def rotate(pts, angle):
    '''
    Apply a rotation to an array of points. Inside of the function you will need to 
    create the appropriate rotation matrix, and then perform the requisite matrix 
    multiplication, and return the result.
    Inputs:
        pts -- a (2,n) array of points, x-coordinates on the first row, y-coordinates
                on the second row.
        angle -- a scalar value indicating the angle of rotation in radians.
    Returns:
        a (2,n) arrary representing the rotated points.
    '''
    a = math.cos(angle)
    b = math.sin(angle)
    rot = np.array([a,-b,b,a]).reshape((2,2))
    return np.dot(rot, pts)


def shear(pts, c, direction):
    '''
    Apply a shear to an array of points. You will need an if statement to deal with
    the possible values of the direction argument.
    Inputs:
        pts -- a (2,n) array of points, x-coordinates on the first row, y-coordinates
                on the second row.
        c -- a scalar value indicating the amount of shearing
        direction -- value of 0 means horizontal shear, 1 means vertical shear.
    Returns:
        a (2,n) array holding the sheared points.
    '''
    shear_mat = np.eye(2)
    if direction==0:
        shear_mat[0,1]=c
    else:
        shear_mat[1,0]=c
    return np.dot(shear_mat,pts)


def reflect(pts, axis):
    '''
    Apply a reflection to an array of points. You will need to use this array to create 
    the appropriate reflection matrix, and then perform matrix multiplication.
    Inputs:
         pts -- a (2,n) array of points, x-coordinates on the first row, y-coordinates
                on the second row.
         axis -- a length two array representing a vector pointing in the direction 
                  of the axis of rotation.
    Returns:
        a (2,n) array containing the reflected points.
    '''
    a = ((axis**2)*np.array([1,-1])).sum()
    b = 2*axis.prod()
    c = (axis**2).sum()
    ref = (np.array([a,b,b,-a]).reshape((2,2)))/float(c)
    return np.dot(ref,pts)


def translate(pts, shift):
    '''
    Apply a translation to an array of points. To perform the translation, you simply 
    need to add the shift array to the pts array, but before doing this, use the 
    .reshape method on the shift array to ensure that it has the same shape as the 
    example given in the text.
    Inputs:
        pts -- a (2,n) array of points, x-coordinates on the first row, y-coordinates
                on the second row.
        shift -- a length two array representing the amount of translation in each direction.
    '''
    return pts + shift.reshape((2,1))


def trajectory(t,w,v,s):
    '''
    Calculate the trajectory of an object rotating about a movign point. 
    Hint: you will use a composition of rotation and translation in this problem. 
    Make sure the function returns an array indicating the new position of p1 at 
    time t. 
    Inputs:
        t -- time (in seconds)
        w -- angular velocity of object (radians/seconds)
        v -- direction of the center of rotation
        s -- speed of center of rotation (meters/second)
    Returns:
        the position of the rotating object at time t.
    '''
    angle = t*w
    p = np.array([[0],[1]])
    shift = ((s*t)/la.norm(v))*v
    return translation(rotation(p, angle), shift)


def plotTrajectory():
    '''
    Plot the trajectory of the rotating particle as calculated in the trajectory
    function. 
    '''
    v = np.array([1, 1])
    times = np.arange(0,10,.1)
    pos = np.zeros((2,len(times)))
    for i in range(len(times)):
        pos[:,i] = trajectory(times[i],np.pi,v,3).flatten()
    plt.plot(pos[0], pos[1])
    plt.show()
    
