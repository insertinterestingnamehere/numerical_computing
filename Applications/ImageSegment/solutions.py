'''
Solutions file for Volume 1 Lab 16
'''

# Remember to include all necessary imports here
import numpy as np
import scipy.sparse as spar
from scipy import linalg as la
from scipy.sparse import linalg as sparla
import matplotlib.pyplot as plt

def laplacian(A):
    '''
    Compute the Laplacian matrix of the adjacency matrix A,
    as well as the second smallest eigenvalue.
    Inputs:
        A -- adjacency matrix for undirected weighted graph,
             shape (n,n)
    Returns:
        L -- the Laplacian matrix of A
        lambda -- second smallest eigenvalue of L
    '''
    # calculate the degree of each vertex. Doesn't matter which axis
    D = A.sum(axis=1)
    # calculate the laplacian matrix
    L = np.diag(D) - A
    # calculate the eigenvalues, sort them
    e = np.real(la.eig(L, right=False))
    e.sort()
    return L, e[1]

def adjacency(img, radius=5.0, sigmaI = .02, sigmaX = 3.0):
    '''
    Compute the weighted adjacency matrix for
    the image array img given the weights and radius. Make sure
    the computations deal with sparse matrices, and make sure you
    return a sparse matrix. Also return an array giving the
    main diagonal of the degree matrix.
    Inputs:
        img -- numpy array representing the image
        radius -- floating point number
        sigmaI -- floating point number
        sigmaD -- floating point number
    Returns:
        W -- the weighted adjacency matrix of img, in sparse form.
        D -- 1D array representing the main diagonal of the degree matrix.
    '''
    # Here are the first steps.
    nodes = img.flatten()
    height, width = img.shape
    W = spar.lil_matrix((nodes.size, nodes.size), dtype=float)
    D = np.zeros((1, nodes.size))

    # Now you do the rest. You need to initialize the elements of W.
    # Remember, since W is sparse, only initialize elements that are nonzero.

    for row in xrange(height):
        for col in xrange(width):
            # Calculate the index of the pixel at (row, col) relative to the
            # flattened array nodes. Hint: row*width + col
            rowcol = row * width + col
            
            # find the indices and distancess of the pixels that are within 
            # distance r of the current pixel by calling getNeighbors
            nbrs = getNeighbors(row, col, radius, height, width)
            
            # calculate the weights corresponding to each pixel and the current
            # pixel. This may be done in a vectorized fashion.
            weights = np.exp(-np.abs(nodes[nbrs[0]] - nodes[rowcol])/sigmaI - nbrs[1]/sigmaX)
            W[rowcol, nbrs[0]] = weights
            D[0,rowcol] = weights.sum()
   
    # Convert W into csc format using the command below.
    # this format is better for computations, while the lil format is better for
    # building the matrix.
    W = W.tocsc()
    return W, D

def getNeighbors(row, col, radius, height, width):
    '''
    Calculate the indices and corresponding distances of pixels within radius
    of the pixel at (row,col), where the pixels are in a (height, width) shaped
    array. The returned indices are with respect to the flattened version of the
    array. This is a helper function for adjacency.
    Inputs:
        row, col -- denotes the row and column number of the pixel we are 
                    centered at.
        radius -- radius of the circular region centered at pixel (row, col)
        height, width -- the height and width of the original image, in pixels
    Returns:
        indices -- a flat array of indices of pixels that are within distance r
                   of the pixel at (row, col)
        distances -- a flat array giving the respective distances from these 
                     pixels to the center pixel.
    '''
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r+1, width))
    y = np.arange(max(row - r, 0), min(row + r+1, height))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(((X-np.float(col))**2+(Y-np.float(row))**2))
    mask = (R<radius)
    return (X[mask] + Y[mask]*width, R[mask])

def segment(img):
    '''
    Compute two segments of the image as described in the text.
    Use your adjacency function to calculate W and D.
    Compute L, the laplacian matrix.
    Then compute D^(-1/2)LD^(-1/2), and find the eigenvector
    corresponding to the second smallest eigenvalue.
    Use this eigenvector to calculate a mask that will be used
    to extract the segments of the image.
    Inputs:
        img -- image array of shape (n,m)
    Returns:
        seg1 -- an array the same size as img, but with 0's
                for each pixel not included in the positive
                segment (which corresponds to the positive
                entries of the computed eigenvector)
        seg2 -- an array the same size as img, but with 0's
                for each pixel not included in the negative
                segment.
    '''
    # call the function adjacency to obtain the adjacency matrix W
    # and the degree array D. 
    W,D = adjacency(img)
    
    # calculate D^(-1/2)
    Dsq = np.sqrt(1.0/D)
    
    # convert D and D^(-1/2) into diagonal sparse matrices (format = 'csc')
    Ds = spar.spdiags(D, 0, D.shape[1], D.shape[1], format = 'csc')
    Dsqs = spar.spdiags(Dsq, 0, D.shape[1], D.shape[1], format = 'csc')
    
    # calculate the Laplacian, L
    L = Ds - W
    
    # calculate the matrix whose eigenvalues we will compute, D^(-1/2)LD^(-1/2)
    # np.dot will not work on sparse arrays. Instead, if P and Q are sparse
    # matrices that we would like to multiply, use P.dot(Q)
    P = Dsqs.dot(L.dot(Dsqs))
    
    # calculate the eigenvector. Use the eigs function in sparla. 
    # Be sure to set the appropriate keyword argument so that you 
    # compute the two eigenvalues with the smallest real part.
    e = sparla.eigs(P, k=2, which="SR")
    eigvec = e[1][:,1]
    
    # create a mask array that is True wherever the eigenvector is positive.
    # reshape it to be the size of img.
    mask = (eigvec>0).reshape(img.shape)
    
    # create the positive segment by masking out the pixels in img 
    # belonging to the negative segment.
    pos = img*mask
    
    # create the negative segment by masking out the pixels in img 
    # belonging to the posative segment.
    neg = img*~mask
    
    # return the two segments (positive first)
    return pos, neg
    
def passoff():
    '''
    I will run this function to pass you off. It calls your
    other functions to generate and plot the segments
    of the image dream.png.
    '''
    print 'in passoff'
    # import the image dream.png, then convert it to grayscale
    img_color = plt.imread('dream.png')
    img = (img_color[:,:,0]+img_color[:,:,1]+img_color[:,:,2])/3.0

    # now calculate and plot the two segments.
    print 'segmenting'
    pos, neg = segment(img)
    print 'plotting...'
    # plot the two segments together with the original.
    plt.subplot(131)
    plt.imshow(neg)
    plt.subplot(132)
    plt.imshow(pos)
    plt.subplot(133)
    plt.imshow(img_color)
    plt.show()
    plt.clf()
    
passoff()
