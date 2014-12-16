import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import svd, norm

def svd_approx(A, k):
    '''
    Calculate the best rank k approximation to A with respect to the induced
    2-norm. Use the SVD.
    Inputs:
        A -- array of shape (m,n)
        k -- positive integer
    Returns:
        Ahat -- best rank k approximation to A obtained via the SVD
    '''
    #compute the reduced SVD
    U,s,Vh = svd(A,full_matrices=False)
    
    #keep only the first k singular values
    S = np.diag(s[:k])
    
    #reconstruct the best rank k approximation
    return U[:,:k].dot(S).dot(Vt[:k,:])
    
    
def lowest_rank_approx(A,e):
    '''
    Calculate the lowest rank approximation to A that has error stricly less than e.
    Inputs:
        A -- array of shape (m,n)
        e -- positive floating point number
    Returns:
        Ahat -- the best rank s approximation of A constrained to have error less than e, 
                where s is as small as possible.
    '''
    #calculate the reduced SVD
    U,s,Vh = svd(A,full_matrices=False)
    
    #find the index of the first singular value less than e
    k = np.where(s<e)[0][0] 
    
    #now recreate the rank k approximation
    S = np.diag(s[:k])
    return U[:,:k].dot(S).dot(Vt[:k,:])
    

def readimg(filename, channel=None):
    if channel is not None:
        return sp.misc.imread(filename)[:,:,channel]
    else:
        return sp.misc.imread(filename)


def compressSVD(filename, rank, random=False, channel=None):
    img = readimg(filename, channel)

    try:
        isize = img[:,:,0].shape
        colors = [la.svd(img[:,:,i]) for i in range(3)]
    except IndexError:
        isize = img.shape
        plt.gray()
        colors = la.svd(img)

    plt.ion()
    imgc = plt.imshow(img)
    newimg = sp.zeros_like(img)

    rank = range(1,rank+1)

    if random is True:
        sp.random.shuffle(rank)

    for r in rank:
        col_res = hat(colors, r-1, r)
        try:
            #col_res[0] = sp.where(col_res[0]>255, col_res[0], 255)
            #col_res[1] = sp.where(col_res[1]>255, col_res[1], 255)
            #col_res[2] = sp.where(col_res[2]>255, col_res[2], 255)

            #col_res[0] = sp.where(col_res[0]<0, col_res[0], 0)
            #col_res[1] = sp.where(col_res[1]<0, col_res[1], 0)
            #col_res[2] = sp.where(col_res[2]<0, col_res[2], 0)
            
            newimg[:,:,0] += col_res[0]
            newimg[:,:,1] += col_res[1]
            newimg[:,:,2] += col_res[2]

            ## for ch in range(3):
            ##     newimg[newimg[:,:,ch]<1]=0
            ##     newimg[newimg[:,:,ch]>254]=255
                
        except IndexError:
            newimg += col_res[0]
            ## newimg[newimg<1]=0
            ## newimg[newimg>254]=255
        
        imgc.set_data(newimg)
        plt.draw()
    plt.ioff()
    plt.show()

    return newimg


def hat(color_svd, lrank, urank):
    results = []
    if len(color_svd) == 3:
        r = 3
    else:
        r = 1
        
    for c in range(r):
        U = color_svd[c][0]
        S = sp.diag(color_svd[c][1])
        Vt = color_svd[c][2]
        results.append(U[:,lrank:urank].dot(S[lrank:urank, lrank:urank]).dot(Vt[lrank:urank,:]))
        
    return results
    
def matrix_rank(X):
    """Compute the rank of a matrix using the SVD"""
    
    S = la.svd(X, compute_uv=False)
    tol = S.max()*sp.finfo(S.dtype).eps
    return sum(S>tol)
