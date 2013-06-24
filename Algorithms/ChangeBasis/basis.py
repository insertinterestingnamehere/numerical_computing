import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

x=[-1.5,-1.,-.5,0.,.5,1.,1.5,.75,-.75]
y=[0.,-1.,-2.,-2.,-2.,-1.,0.,2.,2.]

A=np.array([x,y])

def strc(A,x,y):
    strec=np.array([[x,0],[0,y]])
    newA=sp.dot(strec,A)
    plt.subplot(2,1,1)
    plt.plot(A[0],A[1],'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.subplot(2,1,2)
    plt.plot(newA[0],newA[1],'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()

strc(A,2,1.5)

def rotate(A,ang):
    rotat=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    newA=sp.dot(rotat,A)
    plt.subplot(2,1,1)
    plt.plot(A[0],A[1],'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.subplot(2,1,2)
    plt.plot(newA[0],newA[1],'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()

ang=3*np.pi/16.
rotate(A,ang)

def shift(A,x,y):
    shift=np.array([[x],[y]])
    newA=A+shift
    plt.subplot(2,1,1)
    plt.plot(A[0],A[1],'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.subplot(2,1,2)
    plt.plot(newA[0],newA[1],'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()

shift(A,2,1.5)

def combo(A,strX,strY,ang,shiftX,shiftY):
    strec=np.array([[strX,0],[0,strY]])
    rotat=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    shift=np.array([[shiftX],[shiftY]])
    newA=sp.dot(sp.dot(rotat,strec),A)+shift
    plt.subplot(2,1,1)
    plt.plot(A[0],A[1],'.')
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.subplot(2,1,2)
    plt.plot(newA[0],newA[1],'.')
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.show()

ang=3*np.pi/4.
combo(A,2.,2.,ang,1.,-2.)

def imgR(img,ang):
    m=img.shape[0]
    n=img.shape[1]
    q=int(max(m,n)*1.5)
    newImg=np.ones((q,q,3))
    for i in xrange(m):
        for j in xrange(n):
            k=int(round((i-m/2)*np.cos(ang)+(j-n/2)*-np.sin(ang)+q/2))
            l=int(round((i-m/2)*np.sin(ang)+(j-n/2)*np.cos(ang)+q/2))
            newImg[k,l,:]=img[i,j,:]
    plt.imshow(newImg)
    plt.show()

img_color = plt.imread('dream.png')

ang=2*np.pi/8.
imgR(img_color[:,:],ang)

plt.imshow(img_color)
plt.show()