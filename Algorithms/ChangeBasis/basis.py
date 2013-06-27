import numpy as np
from numpy import math
from matplotlib import pyplot as plt

x = [-1.5, -1., -.5, 0., .5, 1., 1.5, .75, -.75]
y = [0., -1., -2., -2., -2., -1., 0., 2., 2.]

A = np.array([x, y])


def strc(A, x, y):
    strec = np.array([[x, 0.], [0., y]])
    newA = np.dot(strec, A)

    plt.subplot(2, 1, 1)
    plt.plot(A[0], A[1], '.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.subplot(2, 1, 2)
    plt.plot(newA[0], newA[1], '.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


def rotate(A, ang):
    c, s = math.cos(ang), math.sin(ang)

    rotat = np.array([[c, -s], [s, c]])
    newA = np.dot(rotat, A)
    
    plt.subplot(2, 1, 1)
    plt.plot(A[0], A[1], '.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.subplot(2, 1, 2)
    plt.plot(newA[0], newA[1], '.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


def shift(A, x, y):
    shift = np.array([[x], [y]])
    newA = A+shift
    
    plt.subplot(2, 1, 1)
    plt.plot(A[0], A[1], '.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.subplot(2, 1, 2)
    plt.plot(newA[0], newA[1], '.')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


def combo(A, strX, strY, ang, shiftX, shiftY):
    c, s = math.cos(ang), math.sin(ang)

    strec = np.array([[strX, 0.], [0., strY]])
    rotat = np.array([[c, -s], [s, c]])
    shift = np.array([[shiftX], [shiftY]])
    newA = rotat.dot(strec).dot(A)+shift

    plt.subplot(2, 1, 1)
    plt.plot(A[0], A[1], '.')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.subplot(2, 1, 2)
    plt.plot(newA[0], newA[1], '.')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()



def imgR(img, ang):
    m, n = img.shape[:2]
    c, s = math.cos(ang), math.sin(ang)

    q = int(max(m, n)*1.5)
    newImg = np.ones((q, q, 3))
    for i in xrange(m):
        for j in xrange(n):
            k = int(round((i-m/2)*c+(j-n/2)*-s+q/2))
            l = int(round((i-m/2)*s+(j-n/2)*c+q/2))
            newImg[k, l,:] = img[i, j,:]
    plt.imshow(newImg)
    plt.show()

def rotImg():
    img_color = plt.imread('dream.png')

    ang = 2*np.pi/8.
    imgR(img_color[:,:], ang)


    plt.imshow(img_color)
    plt.show()

if __name__ == "__main__":
    strc(A, 2, 1.5)

    ang = 3*np.pi/16.
    rotate(A, ang)

    shift(A, 2, 1.5)

    ang = 3*np.pi/4.
    combo(A, 2., 2., ang, 1., -2.)

    rotImg()
