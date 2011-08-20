import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as la

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
