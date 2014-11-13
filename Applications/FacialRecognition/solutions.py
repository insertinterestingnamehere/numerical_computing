##################################
#  Solutions to Current Version  #
##################################
import numpy as np
from scipy.linalg import norm, svd
from os import walk
from scipy.ndimage import imread
from matplotlib import pyplot as plt
import matplotlib.cm as cm 
from random import sample

# convenience function for plotting flattened grayscale images of width 200, height 180
def show(im, w=200, h=180):
    plt.imshow(im.reshape((w,h)), cmap=cm.Greys_r)
    plt.show()
def show2(im1, im2, w=200, h=180):
    plt.subplot(121)
    plt.imshow(im1.reshape((w,h)), cmap=cm.Greys_r)
    plt.subplot(122)
    plt.imshow(im2.reshape((w,h)), cmap=cm.Greys_r)
    plt.show()

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

F = getFaces()

# we now calculate and plot the mean face
mu = F.mean(axis=1)
show(mu)

# shift each face by the mean, show an example of a mean-shifted face
A = F - mu.reshape((len(mu), 1))
show(A[:,0])

# calculate svd of A
U,Sig,Vh = svd(A, full_matrices=False)

# show one of the eigenfaces
show(U[:,10])

def nEigenfaces(U, A, n):
    """
    Return U_n and Ahat_n as described in lab.
    """
    Un = U[:,:n]
    return Un, Un.T.dot(A)

# project the face vectors onto an eigenface subspace, plot one of them
n_faces = 38
Un, Ahat = nEigenfaces(U,A,n_faces)

def findNearest(M, query):
    """
    Return the index of the column of M that is closest to g wrt Euclidean distance.
    """
    m = M.shape[0]
    return np.argmin(np.linalg.norm(M-query.reshape((m,1)), axis=0))
    
# let's gather some random images from the directory, and try to recognize them!
n_tests = 10
test_files = []
for (dirpath, dirnames, filenames) in walk("./faces94"):
    for f in filenames:
        if f[-3:]=="jpg": # only get jpg images
            test_files.append(dirpath+"/"+f)
test_files = sample(test_files, n_tests)
test_images = np.array([imread(f).mean(axis=2).ravel() for f in test_files]).T

# here is the recognition part
for i in xrange(n_tests):
    ghat = U[:,:n_faces].T.dot(test_images[:,i]-mu)
    ind = findNearest(Ahat, ghat)
    show2(test_images[:,i], F[:,ind])


##################################
#    Solutions to Old Version    #
##################################

import re #For regular expressions

import numpy as np
from os import walk
from scipy.ndimage import imread
import numpy.linalg as la


from scipy.spatial import KDTree

from sklearn import neighbors

import matplotlib.pyplot as plt
import matplotlib.cm as cm


####################################
#       Supporting Functions       #
####################################


def getImages(filePath,fileStep=1):
    """
    Get every fileStep-th image in the directory filePath
    """
    
    #Walk through the directory, creating a list of
    #All the files
    files = []
    for (dirpath,dirnames, filenames) in walk(filePath):
        files.extend([ dirpath +"/" + file for file in filenames])
        
    #Sort the files, for consistency
    files.sort()
    
    images = []
    
    #Use a regular expression to get the jpg files
    #Since not all the files are images
    regex = re.compile(r".*\.jpg")
    for file in files[::fileStep]:
        #Only add images
        if(regex.match(file)):
            images.append(imread(file,flatten=1))
            
    #Return the dimensions of the image and the array of images
    return np.array(images),images[0].shape

    
def showImage(image):
    """
    Show n x n grid as a grayscale image
    Such as the images of the faces
    """
    plt.imshow(image,cm.gray)


####################################
#     Facial Recognition Class     #
####################################

class FacialRec:
    
    """ 
    Represents a Facial Recognition Database
    which can be initialized with a directory
    of training faces and then queried to match
    new faces to those in the database
    """
    
    
    #   All the members of FacialRec:
    #   Let w,h be the width and height of the image
    #   and f be the number of images (or faces) in the database
    
    #   images                  The images in the database
    #                           f x h x w np.array
    
    #   meanImage               The mean image/face of the database
    #                           h x w np.array
    
    #   differenceFaces         The images minus the mean face
    #                           f x w x h array of 
    
    
    #   differenceVectors    
    #   eigenfaces
    #   kdtree
    #   searchOrder
    
    
    def __init__(self,filePath,fileStep,searchOrder=None):
        """
        Constructor of Facial Recognition. You shouldn't have to change
        anything here, but you do need to implament each of the functions
        below
        """
        self.searchOrder = searchOrder
        
        self.initImages(filePath,fileStep)
        self.initMeanImage()
        self.initDifferences()
        self.initEigenFaces()
        self.initClassifier()
    
    def initImages(self,filePath,fileStep):
        """
        Get the images from the given file path, skipping "filestep" 
        files each time
        """
        
        self.images , self.dim = getImages(filePath,fileStep)
        
    def initMeanImage(self):
        """
        Set self.meanImage as the pixel average of the images in the 
        database self.meanImage should be a h x w array (self.images 
        is already f x h x w
        
        Hint: use np.sum to do this without a for loop, make sure
        to give it the right axis
        """
        self.meanImage = (np.sum(self.images,axis=0)*1.0)/(len(self.images))
        
    def initDifferences(self):
        """
        First set self.differenceFaces as the images in the database
        minus the meanImage.
        
        Then use reshape to set differenceVectors to a matrix with it's
        rows as the flattened differenceFaces.
        
        self.differenceVectors should be a f x (hw) (that's 2d) np.array
        """
        
        #Find the difference faces by subtracting the meanImage from each image
        self.differenceFaces = self.images-self.meanImage
        
        #Store the faces as vectors of points in one big matrix
        self.differenceVectors = self.differenceFaces.reshape(
            self.differenceFaces.shape[0],
            self.differenceFaces.shape[1]*self.differenceFaces.shape[2])
        
    def initEigenFaces(self):
        """ 
        Use the SVD to commpute the eigenvectors of the covariance matrix
        self.differenceVectors X self.differenceVectors.T
        
        Set self.eigenFaces to have the normalized eigenvectors of the 
        covariance matrix as its rows
        
        self.eigenfaces should be a f x (hw)  np.array
        
        Note: scipy.linalg.svd returns u,s,v.T. The columns of v are already 
        normalized, but be careful with the orientation of v
        """
        
        u,s,vT = la.svd(self.differenceVectors,full_matrices=False)
        self.eigenFaces = vT
        
    
    def projectToImageFaceSpace(self,image):
        """
        Project image  (a h x w np.array) into the the space of the eigenface. 
        First you must prepare the image by subracting off the mean image then 
        flattening the  difference. You can then compute the inner product of
        this difference  vector with the eigenfaces matrix for the coefficients 
        of the linear combination of eigenfaces representing image.
        
        The output should be a length f np.array
        """
        
        #Zero the image by subracting the database's mean image
        diffFace = image - self.meanImage
        
        #Convert the result to a one dimensional vector
        diffVec = diffFace.ravel()
        
        #Inner product this vector with the eigenfaces for its projection $
        #in the eigenface space
        return np.inner(self.eigenFaces,diffVec)
    
    def rebuildFromEigenFaces(self,coefs,k=None):
        """
        Rebuild the image from the coefficients (coefs) of the linear 
        combination of the eigenfaces, but only using a k number of coefficients
        /eigenfaces. To do this, you simply compute this linear combination.
        
        This can be done using a single matrix operation between coefs and the 
        eigenfaces, taking the correct array slices to limit the combination to
        the first k coefficients. Reshape the result to self.dim.
        Don't forget to add back in the mean image you subtracted off earlier
        
        The output should have the same dimensions as the original image, h x w
        
        Also, if k is set to none, set k to be the number of eigenFaces
        """
        
        #Free Code!
        #if k == None:
            #k = self.eigenFaces.shape[0]
            
            
        return (self.meanImage + np.dot(coefs[:k],self.eigenFaces[:k,:])
                                    .reshape(self.dim))
    
    def initClassifier(self):
        """
        First, create a list of training points by projecting each image in the 
        database into the eigenface basis.
        
        This should be a f x f array f length list of f length arrays, with each
        row a different image in the eigenface basis
        
        Then create a KNeighborsClassifier with n_neighbors=1 and set self.nbrs
        to this. Then fit the trainingPoints, but using an array slice to only 
        use the first k coefficients to match ( this will cut out part of the
        2nd dimension of training points). Use the indicies of the training 
        points (you can use an xrange for this) as the labels for self.nbrs.fit
        
        Also, if the searchOrder is set to None, set it to the total number of
        eigenfaces
        """
        
        #Free Code!
        #if(self.searchOrder==None):
            #self.searchOrder = self.eigenFaces.shape[0]
            
        #Compute the training points  
        trainingPoints = np.array(
            [self.projectToImageFaceSpace(image) for image in self.images]) 
        
        #Create the classifier
        self.nbrs = neighbors.KNeighborsClassifier(n_neighbors = 1)
        
        #Fit it to the training points, using the incidies as labels
        self.nbrs.fit(trainingPoints[:,:self.searchOrder],xrange(trainingPoints.shape[0]))
        
    def findNearest(self,image):
        """
        perform a nearst neighbor search to find the image in the database to
        find the closest match to image in the database. Return the matching
        image in the database and the euclidean distance between the match and 
        image in the eigenface basis
        
        Do this by first projecting the image into the face space then using 
        nbrs.predict to find the index of the matching image.
        
        To compute the distance, project the matching image into eigenface 
        basis, then take the np.norm of the difference between the matching
        coefficients and the image coefficients
        
        match should be a h x w np.array, the distance a scalar
        """
        imageCoefs = self.projectToImageFaceSpace(image)
        
        i = self.nbrs.predict(imageCoefs[:self.searchOrder])
        
        match = self.images[i]
        
        
        #compute the distance between the two
        matchCoefs = self.projectToImageFaceSpace(match)
        
        dist = la.norm(matchCoefs - imageCoefs)
        
        return match, dist
    
    
####################################
#   Fucntions to test FacialRec    #
####################################

def checkMeanImage(faces):
    """
    Plot the mean face of a facial recognition database
    """
    
    showImage(faces.meanImage)
    plt.title("Mean Face")
    plt.show()
   
def checkDifferenceFaces(faces):
    """
    Plot the first 3 difference faces of a facial recognition database
    """
    for i in range(3):
        plt.subplot(130+i)
        showImage(faces.differenceFaces[i])
    plt.title("Difference Faces")
    plt.show()
        
def checkEigenFaces(faces):
    """
    Plot the first 3 eigenfaces of a facial recognition database
    """
    for i in range(3):
        plt.subplot(130+i)
        showImage(faces.eigenFaces[i].reshape(faces.dim))
    plt.title("Eigenfaces")
    plt.show()
  
def checkRebuildImage(faces,image):
    """
    Rebuild an image with varying number of eigenfaces
    """
    coefs = faces.projectToImageFaceSpace(image)
    
    plt.subplot(130)
    showImage(faces.rebuildFromEigenFaces(coefs))
    
    plt.subplot(131)
    showImage(faces.rebuildFromEigenFaces(coefs,coefs.size//2))
    
    plt.subplot(132)
    showImage(faces.rebuildFromEigenFaces(coefs,coefs.size//4))
    
    plt.title("Rebuilt faces with full, half, and fourth of the coefficients")
    plt.show()
