import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np

import solutions as sol

from scipy.misc import imsave


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
    faces = sol.FacialRec("./faces94/male/",30)
    images,dim = sol.getImages("./faces94/male/")
    
    checkMeanImage(faces)
    checkDifferenceFaces(faces)
    checkEigenFaces(faces)
    checkRebuildImage(faces,images[0])