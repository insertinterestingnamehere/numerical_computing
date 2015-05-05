import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
from scipy.misc import imsave
from scipy.linalg import svd
from os import walk
from scipy.ndimage import imread

def getFaces(path="./faces94"):
    # these are the dimensions of the images
    w = 200
    h = 180
    
    # traverse the directory, get one image per subdirectory
    faces = []
    for (dirpath, dirnames, filenames) in walk(path):
        for f in filenames:
            if f[-3:]=="jpg": # only get jpg images
                # load image, convert to grayscale, flatten into vector
                face = imread(dirpath+"/"+f).mean(axis=2).ravel() 
                faces.append(face)
                break
                
    # put all the face vectors column-wise into a matrix
    F = np.array(faces).T
    return F

def plotMeanFace(mu):
    imsave("meanFace.png", mu.reshape((200, 180)))

def plotDiffFaces(A):
    for j, i in enumerate([0,20,50]):
        imsave("differenceFace{0}.png".format(j),A[:,i].reshape((200,180)))

def plotEigenFaces(U):
    for i in xrange(3):
        imsave("eigenface{0}.png".format(i),U[:,i].reshape((200,180)))

def plotReconstructed(U, A, mu):
    imsave("rebuiltAll.png", (A[:,20]+mu).reshape((200,180)))
    s = U.shape[1]
    r1 = U[:,:75].dot(U[:,:75].T.dot(A[:,0])) + mu
    r2 = U[:,:38].dot(U[:,:38].T.dot(A[:,0])) + mu
    r3 = U[:,:19].dot(U[:,:19].T.dot(A[:,0])) + mu
    r4 = U[:,:9].dot(U[:,:9].T.dot(A[:,0])) + mu
    r5 = U[:,:5].dot(U[:,:5].T.dot(A[:,0])) + mu
    imsave("rebuiltHalf.png", r1.reshape((200,180)))
    imsave("rebuiltFourth.png", r2.reshape((200,180)))
    imsave("rebuiltEighth.png", r3.reshape((200,180)))
    imsave("rebuiltSixteenth.png", r4.reshape((200,180)))
    imsave("rebuiltThirtySecond.png", r5.reshape((200,180)))
  
    
if __name__ == "__main__":
    F = getFaces(path="../../../faces94")
    print F.shape
    mu = F.mean(axis=1)
    A = F - mu.reshape((len(mu), 1))
    U,Sig,Vh = svd(A, full_matrices=False)
    plotMeanFace(mu)
    plotDiffFaces(A)
    plotEigenFaces(U)
    plotReconstructed(U, A, mu)
