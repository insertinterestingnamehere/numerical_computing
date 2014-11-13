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
    r1 = U[:,:s/2].dot(U[:,:s/2].T.dot(A[:,20])) + mu
    r2 = U[:,:s/4].dot(U[:,:s/4].T.dot(A[:,20])) + mu
    imsave("rebuiltHalf.png", r1.reshape((200,180)))
    imsave("rebuiltFourth.png", r2.reshape((200,180)))

# below are the plotting functions from a previous version
def checkMeanImage(faces):
    """
    Plot the mean face of a facial recognition database
    """
    
    imsave("meanFace.png",faces.meanImage)
   
def checkDifferenceFaces(faces):
    """
    Plot the first 3 difference faces of a facial recognition database
    """
    for i in range(3):
        imsave("differenceFace{0}.png".format(i),faces.differenceFaces[i])
        
def checkEigenFaces(faces):
    """
    Plot the first 3 eigenfaces of a facial recognition database
    """
    for i in range(3):
        imsave("eigenface{0}.png".format(i),faces.eigenFaces[i].reshape(faces.dim))


def checkRebuildImage(faces,image):
    """
    Rebuild an image with varying number of eigenfaces
    """
    coefs = faces.projectToImageFaceSpace(image)
    
    imsave("rebuiltAll.png",faces.rebuildFromEigenFaces(coefs))
    
    imsave("rebuiltHalf.png",faces.rebuildFromEigenFaces(coefs,coefs.size//2))
    
    imsave("rebuiltFourth.png",faces.rebuildFromEigenFaces(coefs,coefs.size//4))
    
    
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
