import scipy as sp
from scipy import linalg as la

def dct_mat(n):
    """Construct an nxn DCT matrix"""
    tmp = sp.array(sp.cos([(j*.5)*k*sp.pi/n for j in range(n) for k in range(n)]).reshape((n,n))).T
    return sp.array([i/la.norm(i) for i in tmp])

def dct1d_expr(data, blksize, npoints=200, terms=-1):
    """Interpolate single dimensional data using the previously calculated DCT coefficients"""
    data = data.astype(float)#s.flatten()

    n = float(len(data))
    x_points = sp.linspace(0,n,float(npoints))

    n = len(data)
    #xy = sp.math.sqrt(npoints)
    #precalculate some often-used values
    sqrt1n = sp.math.sqrt(1.0/n)
    sqrt2n = sp.math.sqrt(2.0/n)
    n2 = 2.0*n

    tmp = lambda t: sp.sum([co*sp.math.cos((i*sp.pi*(2.0*t+1.0))/n2) for co,i in zip(data[1:terms], sp.arange(1.0,n))])
    y_points = sp.array([sqrt1n*data[0]+sqrt2n*tmp(tvals) for tvals in sp.linspace(0.0, n, npoints)])
    return x_points, y_points
    #return x_points.reshape((n,1)), y_points.reshape((1,n)), z_points.reshape((n,n))

def dct2d_ex(data, blksize, npoints, terms=-1):
    x,y,z1,z2 = None, None, None, None
    for row in data:
        x,z1 = dct1d_expr(row, blksize, npoints, terms)

    for col in data.T:
        y,z2 = dct1d_expr(col, blksize, npoints, terms)

    return x,y,z1,z2

def dct2d_expr(data, npoints, terms=-1):
    n = float(len(data))
    data = data.astype('float32')
    x_points = y_points = sp.linspace(0,n,float(npoints))

    n = len(vec)
    #xy = sp.math.sqrt(npoints)
    #precalculate some often-used values
    sqrt2n = 2.0*sp.math.sqrt(n)
    #sqrt2n = sp.math.sqrt(2.0/n)
    n2 = 2.0*n

    tmp = lambda s, t: sp.sum([sp.sum(co*ak*al*sp.math.cos((i*sp.pi*(2.0*s+1.0))/n2)*sp.math.cos((j*sp.pi*(2.0*t+1.0))/n2)) for co,i,j,ak,al in zip(vec[1:terms])])
    z_points = None


def quantize(DCT, QuantMat=None, inverse=False):
    """Quantize a DCT coefficient matrix.  If QuantMat is None and DCT.shape is (8,8)
    then the standard JPEG quantization matrix is used.

    Otherwise if QuantMat is None then a random integer matrix is generated with values
    in the range of 10...121 (the min and max of the standard JPEG quantization matrix)

    QuantMat must be the same shape as DCT."""
    if QuantMat is None and DCT.shape == (8,8):
        QuantMat = sp.array([[16,11,10,16,24,40,51,61],
                            [12,12,14,19,26,58,60,55],
                            [ 14,13,16,24,40,57,69,56],
                            [ 14,17,22,29,51,87,80,62],
                            [ 18,22,37,56,68,109,103,77],
                            [ 24,35,55,64,81,104,113,92],
                            [ 49,64,78,87,103,121,120,101],
                            [ 72,92,95,98,112,100,103,99]])
    else:
        QuantMat = sp.random.randint(10,121,size=DCT.shape)

    if inverse is True:
        return DCT*QuantMat
    else:
        return DCT/QuantMat

def dct_basis(X, blksize):
    """Calculate DCT coefficients using the basis matrices"""

    bmatr = basis_mats(blksize, plot=False)
    return [sp.dot(X, B) for B in bmatr]

def dct(X, blksize):
    """Perform the DCT transform on X using blksize.
    If X.shape > blksize then X will tiled"""
    
    #Xshape, Yshape = X.shape
    #if Xshape % blksize == 0 and Yshape % blksize == 0:
        #output = sp.zeros(X.shape)
        #for 
    return sp.dot(dct_mat(blksize), X)

def idct(Y, blksize):
    return sp.dot(dct_mat(blksize).T, Y)

def dct2(X, blksize):
    """Calculate DCT transform of a 2D array, X

    In order for this work, we have to split X into blksize chunks"""
    dctm = dct_mat(blksize)

    #try:
    #blks = [sp.vsplit(x, X.shape[1]/blksize) for x in sp.hsplit(X, X.shape[0]/blksize)]
    #except:
    #    print "Some error occurred"

    output = sp.zeros(X.shape)
    if output.ndim==3:
        for i in range(blksize,X.shape[0],blksize):
            for j in range(blksize, X.shape[1], blksize):
                for c in range(X.shape[2]):
                    b = X[i-blksize:i, j-blksize:j, c]
                    output[i-blksize:i, j-blksize:j, c] = sp.dot(sp.dot(dctm,b),dctm.T)
    elif output.ndim==2:
        for i in range(blksize,X.shape[0],blksize):
            for j in range(blksize, X.shape[1], blksize):
                b = X[i-blksize:i, j-blksize:j]
                output[i-blksize:i, j-blksize:j] = sp.dot(sp.dot(dctm,b),dctm.T)
                
    #blks = [sp.dot(sp.dot(dctm, b), dctm.T) for b in blks]
    #return sp.concatenate([blk for blk in blks]).reshape(X.shape)
    return output

def idct2(X, blksize):
    """Calculate the inverse DCT transform of a 2D array, X"""
    dctm = dct_mat(blksize)

    #try:
    #blks = [sp.vsplit(x, X.shape[1]/blksize) for x in sp.hsplit(X, X.shape[0]/blksize)]
    #except:
    #    print "Some error occurred"

    output = sp.zeros(X.shape)
    if output.ndim==3:
        for i in range(blksize,X.shape[0],blksize):
            for j in range(blksize, X.shape[1], blksize):
                for c in range(X.shape[2]):
                    b = X[i-blksize:i, j-blksize:j, c]
                    output[i-blksize:i, j-blksize:j, c] = sp.dot(sp.dot(dctm,b),dctm.T)
    elif output.ndim==2:
        for i in range(blksize,X.shape[0],blksize):
            for j in range(blksize, X.shape[1], blksize):
                b = X[i-blksize:i, j-blksize:j]
                output[i-blksize:i, j-blksize:j] = sp.dot(sp.dot(dctm.T,b),dctm)
    #blks = [sp.dot(sp.dot(dctm.T, b), dctm) for b in blks]
    #return sp.concatenate([blk for blk in blks]).reshape(X.shape)
    return output
    
def basis_mats(blksize=8, plot=False):
    """Compute the basis matrices of the DCT matrix"""

    matr = dct_mat(blksize)
    bmatr = [sp.outer(matr[i], matr[j]) for i in range(blksize) for j in range(blksize)]
    if plot is True:
        ret = sp.zeros((blksize**2, blksize**2))
        for i in range(blksize):
            for j in range(blksize):
                ret[i*blksize:(i+1)*blksize, j*blksize:(j+1)*blksize] = bmatr[i+j*blksize]
        return ret
    else:
        return bmatr
