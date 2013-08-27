import numpy as np
from numpy import math
from matplotlib import pyplot as plt

def plotPoints(oldA, newA):
    plt.subplot(2, 1, 1)
    plt.scatter(oldA[0], oldA[1])
    plt.axis('equal')
    plt.subplot(2, 1, 2)
    plt.scatter(newA[0], newA[1])
    plt.axis('equal')
    plt.show()

def strc(A, scale):
    strec = np.diag(scale)
    newA = np.dot(strec, A)
    return newA

def shear(A, direction, amount):
    L = np.eye(2)
    if (direction == 0):
        L[0][1] = amount
    else:
        L[1][0] = amount
    newA = np.dot(L,A)
    return newA
    
def reflect(A, rAxis):
    a = rAxis[0]**2-rAxis[1]**2
    b = 2*rAxis.prod()
    L = (1.0/(rAxis**2).sum())*np.array([a, b, b, -a]).reshape((2,2))
    newA = np.dot(L,A)
    return newA

def rotate(A, ang):
    c, s = math.cos(ang), math.sin(ang)
    L = np.array([[c, -s], [s, c]])
    newA = np.dot(L, A)
    return newA

def shift(A, offsets):
    newA = A + offsets.reshape((2,1))
    return newA

def trajectory(time, omega, vel):
    ang = time*omega            
    pos = time*vel*np.array([np.sqrt(2)**(-1), np.sqrt(2)**(-1)])
    p0 = np.array([0, 1])   # starting position of the particle
    return shift(rotate(p0,ang), pos)

def combo(A, strX, strY, ang, shiftX, shiftY):
    c, s = math.cos(ang), math.sin(ang)
    strec = np.array([[strX, 0.], [0., strY]])
    rotat = np.array([[c, -s], [s, c]])
    shift = np.array([[shiftX], [shiftY]])
    newA = rotat.dot(strec).dot(A)+shift
    return newA
