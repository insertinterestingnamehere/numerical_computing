#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, exp, fmin, fmax, sqrt
cimport cython

cdef double g(double x, double sigma):
    #The absolute value isn't necessary here
    return exp(-(x/sigma)**2)

def perona_malik3(np.ndarray[dtype=double, ndim=2] u, double lbda, double sigma, int reps):
    cdef int n, i, j
    cdef int height = u.shape[0]
    cdef int width = u.shape[1]

    cdef double temp, dif1, dif2, dif3, dif4
    cdef np.ndarray[dtype=double, ndim=1] temprow = np.empty(width)
    for n in range(reps):
        #python slicing, possibly slow. Maybe do it with a memoryview??
        temprow[:] = u[0]

        for i in range(1, height-1):
            temp = u[i,0]
            for j in range(1, width-1):
                dif1 = temprow[j] - u[i, j]
                dif2 = u[i+1, j] - u[i, j]
                dif3 = temprow[j-1]-u[i, j]
                dif4 = u[i, j+1] - u[i, j]
                temprow[j] = u[i, j]
                u[i, j] += lbda * (g(dif1, sigma) * dif1 + g(dif2, sigma)* dif2 + g(dif3, sigma) * dif3 + g(dif4, sigma) * dif4)

cdef inline double g2(double x, double sinv):
    #in this case the absolute value is not necessary.
    return exp(-(x*sinv)**2)


def perona_malik9(np.ndarray[dtype=double, ndim=2] u, double lbda, double sigma, int reps):
    cdef int n,i,j
    cdef int height = u.shape[0]
    cdef int width = u.shape[1]
    cdef double dif1, dif2, dif3, dif4,
    cdef double sinv = 1./sigma
    cdef np.ndarray[dtype=double, ndim=1] temprow = np.empty(width)

    for n in range(reps):
        #do the first row
        #do the first item in the first row
        dif1 = u[1,0] - u[0,0]
        dif2 = u[0,1] - u[0,0]
        
        temprow[0] = u[0,0]
        u[0,0] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2)
        
        #do the middle entries of the first row
        for j in range(1, width-1):
            dif1 = u[i+1,j] - u[i,j]
            dif2 = temprow[j-1] - u[i,j]
            dif3 = u[i,j+1] - u[i,j]

            temprow[j] = u[i,j] 
            u[i,j] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2 + g2(dif3, sinv) * dif3)

        #do the last entry in the first row
        dif1 = u[1,width-1] - u[0,width-1]
        dif2 = temprow[width-2] - u[0, width-1]
        
        temprow[width-1] = u[0,width-1]
        u[0,width-1] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2)
        
        #do the middle rows
        for i in range(1, height-1):
            #do the first entry of the current row
            dif1 =  temprow[0] - u[i,0]
            dif2 =  u[i+1,0] - u[i,0]
            dif3 =  u[i,1] - u[i,0]
            
            temprow[0] = u[i,0]
            u[i,0] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2 + g2(dif3, sinv) * dif3)
        
            #do the middle entries of the current row
            for j in range(1, width-1):
                dif1 = temprow[j] - u[i,j]
                dif2 = u[i+1,j] - u[i,j]
                dif3 = temprow[j-1] - u[i,j]
                dif4 = u[i,j+1] - u[i,j]
                
                temprow[j] = u[i,j]
                u[i,j] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2 + g2(dif3, sinv) * dif3 + g2(dif4, sinv) * dif4)
        
            #do the last entry of the current row
            dif1 = temprow[width-1] - u[i, width-1]
            dif2 = u[i+1, width-1] - u[i, width-1]
            dif3 = temprow[width-2] - u[i,width-1]

            temprow[width-1] = u[i,width-1]
            u[i,width-1] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2 + g2(dif3, sinv) * dif3)
        
        #do the first entry of the last row
        dif1 = temprow[0] - u[height-1,0]
        dif2 = u[height-1,1] - u[height-1,0]
        
        temprow[0] = u[height-1,0]
        u[height-1,0] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2)
        
        #do the middle entries of the last row
        for j in range(1, width-1):
            dif1 = temprow[j] - u[height-1,j] 
            dif2 = temprow[j-1] - u[height-1,j]
            dif3 = u[height-1,j+1] - u[height-1,j]

            temprow[j] = u[height-1,j]
            u[height-1,j] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2 + g2(dif3, sinv) * dif3)
        
        #do the last entry of the last row
        dif1 = temprow[width-1] - u[height-1,width-1]
        dif2 = temprow[width-2] - u[height-1,width-1]

        #we don't actually need to store the last temp value.
        u[height-1,width-1] += lbda * (g2(dif1, sinv) * dif1 + g2(dif2, sinv) * dif2)


cdef inline double norm(double[:] x, double[:] y):
    #using the infinity norm
    return fmax(fmax(x[0] - y[0], x[1] - y[1]), x[2] - y[2])


def perona_malik_color2(np.ndarray[dtype=double,ndim=3] U, double lbda, double sigma, int reps):
    #this one isn't very readable, but works the same as the black and white version except
    #that instead of a single value for a pixel, now we have three to deal with 3 dimensional vectors
    #we use the norm function defined above for the computation of g at each point.
    cdef int n,i,j,k
    cdef int height = U.shape[0]
    cdef int width = U.shape[1]
    cdef double gg1, gg2, gg3, gg4, temp
    cdef double cst=4./3.
    cdef double sinv=1./sigma
    cdef double[:,:,:] u = U
    cdef double[:,:] temprow = np.empty((width, 3))

    for n in range(reps):
        gg1 = g2(norm(u[1,0], u[0,0]), sinv)
        gg2 = g2(norm(u[0,1], u[0,0]), sinv)
        for k in range(3):
            temprow[0,k] = u[0,0,k]
            u[0,0,k] += lbda * (gg1 * (u[1,0,k] - u[0,0,k]) + gg2 * (u[0,1,k] - u[0,0,k]))
        
        #there may be a bug in this for loop, but I don't see it...
        #the reason is that there appears to be some darkening at the top of the image.
        for j in range(1, width-1):
            gg1 = g2(norm(u[i+1,j], u[i,j]), sinv)
            gg2 = g2(norm(temprow[j-1], u[i,j]), sinv)
            gg3 = g2(norm(u[i,j+1], u[i,j]), sinv)
            for k in range(3):
                temprow[j,k] = u[i,j,k]
                u[i,j,k] += lbda * (gg1 * (u[i+1,j,k] - u[i,j,k]) + gg2 * (temprow[j-1, k]-u[i,j,k]) + gg3 * (u[i,j+1,k] - u[i,j,k]))
        
        gg1=g2(norm(u[1,width-1], u[0,width-1]), sinv)
        gg2=g2(norm(temprow[width-2], u[0,width-1]), sinv)
        for k in range(3):
            temprow[width-1,k] = u[0,width-1,k]
            u[0,width-1,k] += lbda * (gg1 * (u[1,width-1,k] - u[0,width-1,k]) + gg2 * (temprow[width-2,k] - u[0,width-1,k]))
        
        for i in range(1, height-1):
            gg1 = g2(norm(temprow[0], u[i,0]), sinv)
            gg2 = g2(norm(u[i+1,0], u[i,0]), sinv)
            gg3 = g2(norm(u[i,1], u[i,0]), sinv)
            
            for k in range(3):
                temprow[0,k] = u[i,0,k]
                u[i,0,k] += lbda * (gg1 * (temprow[0,k] - u[i,0,k]) + gg2 * (u[i+1,0,k] - u[i,0,k]) + gg3 * (u[i,1,k] - u[i,0,k]))
            
            for j in range(1, width-1):
                gg1 = g2(norm(temprow[j], u[i,j]), sinv)
                gg2 = g2(norm(u[i+1,j], u[i,j]), sinv)
                gg3 = g2(norm(temprow[j-1], u[i,j]), sinv)
                gg4 = g2(norm(u[i,j+1], u[i,j]), sinv)
                for k in xrange(3):
                    temprow[j,k] = u[i,j,k]
                    u[i,j,k] += lbda * (gg1 * (temprow[j,k] - u[i,j,k]) + gg2 * (u[i+1,j,k] - u[i,j,k]) + gg3 * (temprow[j-1,k] - u[i,j,k]) + gg4 * (u[i,j+1,k] - u[i,j,k]))
            
            gg1 = g2(norm(temprow[width-1], u[i,width-1]), sinv)
            gg2 = g2(norm(u[i+1,width-1], u[i,width-1]), sinv)
            gg3 = g2(norm(temprow[width-2], u[i,width-1]), sinv)
            for k in range(3):
                temprow[width-1,k] = u[i,width-1,k]
                u[i,width-1,k] += lbda * (gg1 * (temprow[width-1,k] - u[i,width-1,k]) + gg2 * (u[i+1,width-1,k] - u[i,width-1,k]) + gg3 * (temprow[width-2,k] - u[i,width-1,k]))
        gg1 = g2(norm(temprow[0], u[height-1,0]), sinv)
        gg2 = g2(norm(u[height-1,1], u[height-1,0]), sinv)
        for k in range(3):
            temprow[0,k] = u[height-1,0,k]
            u[height-1,0,k] += lbda * (gg1 * (temprow[0,k] - u[height-1,0,k]) + gg2 * (u[height-1,1,k] - u[height-1,0,k]))
        for j in range(1, width-1):
            gg1 = g2(norm(temprow[j], u[height-1,j]), sinv)
            gg2 = g2(norm(temprow[j-1], u[height-1,j]), sinv)
            gg3 = g2(norm(u[height-1,j+1], u[height-1,j]), sinv)
            for k in range(3):
                temprow[j,k] = u[height-1,j,k]
                u[height-1,j,k] += lbda * (gg1 * (temprow[j,k] - u[height-1,j,k]) + gg2 * (temprow[j-1,k] - u[height-1,j,k]) + gg3 * (u[height-1,j+1,k] - u[height-1,j,k]))
        gg1 = g2(norm(temprow[width-1], u[height-1,width-1]), sinv)
        gg2 = g2(norm(temprow[width-2], u[height-1,width-1]), sinv)
        for k in range(3):
            #we don't actually need to store the last difference
            u[height-1,width-1,k] += lbda * (gg1 * (temprow[width-1,k] - u[height-1,width-1,k]) + gg2 * (temprow[width-2,k] - u[height-1,width-1,k]))

ctypedef np.float32_t float32_T

cdef inline bimin(float32_T[:] arr, int size):
    #changes the first two items of an array to be the
    #two smallest items in the array
    cdef int i
    cdef float32_T min1, min2
    if arr[0] < arr[1]:
        min1, min2 = arr[0], arr[1]
    else:
        min2, min1 = arr[0], arr[1]
    for i in range(2, size):
        if arr[i] < min1:
            min1, min2 = arr[i], min1
        elif arr[1] < min2:
            min2= arr[i]
    arr[0] = min1
    arr[1] = min2


cpdef makecoefs(np.ndarray[dtype=float32_T,ndim=2] vdifs,
                np.ndarray[dtype=float32_T,ndim=2] hdifs, np.ndarray[dtype=float32_T,ndim=2] ldifs,
                np.ndarray[dtype=float32_T,ndim=2] udifs, np.ndarray[dtype=float32_T,ndim=2] coefs):
    #This one isn't particularly well optimized
    #it *shouldn't* be too terrible though.
    #the operations are in place to fill the coeffs array
    #I don't know of any nice way of doing this, so heres a tractable way.
    cdef int i,j,k
    cdef int h = coefs.shape[0]
    cdef int w = coefs.shape[1] 
    cdef float32_T[:] temp = np.empty(8, dtype=np.float32)
    
    #do first point
    temp[0] = vdifs[0,0]
    temp[1] = ldifs[0,0]
    temp[2] = hdifs[0,0]
    bimin(temp, 3)
    coefs[0,0] = (temp[0] + temp[1]) * .5
    for j in range(1, w-1):
        temp[0] = hdifs[0,j-1]
        temp[1] = hdifs[0,j]
        temp[2] = udifs[0,j-1]
        temp[3] = vdifs[0,j]
        temp[4] = ldifs[0,j]
        bimin(temp, 5)
        coefs[0,j] = (temp[0] + temp[1]) * .5
    
    temp[0] = hdifs[0,w-2]
    temp[1] = udifs[0,w-2]
    temp[2] = vdifs[0,w-1]
    bimin(temp, 3)
    coefs[0,0] = (temp[0] + temp[1]) * .5
    for i in range(1,h-1):
        temp[0] = vdifs[i-1,0]
        temp[1] = vdifs[i,0]
        temp[2] = udifs[i-1,0]
        temp[3] = hdifs[i,0]
        temp[4] = ldifs[i,0]
        bimin(temp, 5)
        coefs[i,0] = (temp[0] + temp[1]) * .5
        for j in range(1, w-1):
            temp[0] = ldifs[i-1,j-1]
            temp[1] = ldifs[i,j]
            temp[2] = vdifs[i-1,j]
            temp[3] = vdifs[i,j]
            temp[4] = udifs[i-1,j]
            temp[5] = udifs[i,j-1]
            temp[6] = hdifs[i,j-1]
            temp[7] = hdifs[i,j]
            bimin(temp, 8)
            coefs[i,j] = (temp[0] + temp[1]) * .5
        
        temp[0] = vdifs[i-1,w-1]
        temp[1] = vdifs[i,w-1]
        temp[2] = ldifs[i-1,w-2]
        temp[3] = hdifs[i,w-2]
        temp[4] = udifs[i,w-2]
        bimin(temp, 6)
        coefs[i,w-1] = (temp[0] + temp[1]) * .5
    
    temp[0] = vdifs[h-2,0]
    temp[1] = udifs[h-2,0]
    temp[2] = hdifs[h-1,0]
    bimin(temp, 3)
    coefs[h-1,0] = (temp[0] + temp[1]) * .5
    for j in range(1, w-1):
        temp[0] = hdifs[h-1,j-1]
        temp[1] = hdifs[h-1,j]
        temp[2] = ldifs[h-2,j-1]
        temp[3] = vdifs[h-2,j]
        temp[4] = udifs[h-2,j]
        bimin(temp, 5)
        coefs[h-1,j] = (temp[0] + temp[1]) * .5
    
    temp[0] = hdifs[h-1,w-2]
    temp[1] = ldifs[h-2,w-2]
    temp[2] = vdifs[h-2,w-1]
    bimin(temp, 3)
    coefs[h-1,w-1] = (temp[0] + temp[1]) * .5
