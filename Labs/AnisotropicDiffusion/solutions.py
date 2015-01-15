import numpy as np
import numexpr as ne


#numexpr-ized problem 2
#this one is blazing fast
def perona_malik_ne(u, lbda, sigma, reps):
    #expects array of 32 bit floats
    sinv = np.float32(1./sigma)
    lbda32 = np.float32(lbda)
    unew = u.copy()
    #make some views of the same data as u
    uN = u[:-1]
    uS = u[1:]
    uW = u[:,:-1]
    uE = u[:,1:]
    #make similar views for unew
    uNnew = unew[:-1]
    uSnew = unew[1:]
    uWnew = unew[:,:-1]
    uEnew = unew[:,1:]
    temp = np.empty_like(u)
    vtemp = temp[:-1]
    htemp = temp[:,:-1]
    for i in xrange(reps):
        ne.evaluate('lbda32*exp(-(sinv*(uS-uN))**2)*(uS-uN)', out=vtemp)
        ne.evaluate('uNnew+vtemp', out=uNnew)
        ne.evaluate('uSnew-vtemp', out=uSnew)
        ne.evaluate('lbda32*exp(-(sinv*(uE-uW))**2)*(uE-uW)', out=htemp)
        ne.evaluate('uWnew+htemp', out=uWnew)
        ne.evaluate('uEnew-htemp', out=uEnew)
        u[:] = unew

#problem 2
#vanilla numpy version
#based on diferent views of the same data which
#are made by slicing the arrays involved.
#It is less straightforward, but much shorter.
#This one could probably be better optimized by
#pre-allocation of all needed temporary arrays.
def perona_malik(u, lbda, sigma, reps):
    unew = u.copy()
    #make some views of the same data as u
    uN = u[:-1]
    uS = u[1:]
    uW = u[:,:-1]
    uE = u[:,1:]
    #make similar views for unew
    uNnew = unew[:-1]
    uSnew = unew[1:]
    uWnew = unew[:,:-1]
    uEnew = unew[:,1:]
    temp = np.empty_like(u)
    vtemp = temp[:-1]
    htemp = temp[:,:-1]
    difs = np.empty_like(u)
    vdifs = difs[:-1]
    hdifs = difs[:,:-1]
    for i in xrange(reps):
        vdifs[:] = uS - uN
        vtemp[:] = vdifs / sigma
        np.square(vtemp, out=vtemp)
        vtemp *= -1
        np.exp(vtemp, out=vtemp)
        vtemp *= lbda
        vtemp *= vdifs
        uNnew += vtemp
        uSnew -= vtemp
        hdifs[:] = uE - uW
        htemp[:] = hdifs / sigma
        np.square(htemp, out=htemp)
        htemp *= -1
        np.exp(htemp, out=htemp)
        htemp *= lbda
        htemp *= hdifs
        uWnew += htemp
        uEnew -= htemp
        u[:] = unew


#problem 3
#vanilla numpy version based on array views
def perona_malik_color(u, lbda, sigma, iters):
    unew = u.copy()
    #transposes are necessary so that broadcasting can work later.
    uNT = u[:-1].T
    uST = u[1:].T
    uWT = u[:,:-1].T
    uET = u[:,1:].T
    uNnewT = unew[:-1].T
    uSnewT = unew[1:].T
    uWnewT = unew[:,:-1].T
    uEnewT = unew[:,1:].T
    vdifs = np.empty_like(uNT[0])
    hdifs = np.empty_like(uWT[0])
    temp = np.empty_like(u)
    vtemp = temp[:-1].T
    htemp = temp[:,:-1].T
    temp2 = np.empty_like(temp)
    vtemp2 = temp2[:-1].T
    htemp2 = temp2[:,:-1].T
    for n in xrange(iters):
        #I lined the computations up in this way so as to 
        #avoid unnecessary memory allocation and deallocation
        #as we go through the for loop.
        vtemp[:] = uST - uNT
        np.fabs(vtemp, out=vtemp2)
        np.max(vtemp2, axis=0, out=vdifs)
        vdifs /= sigma
        np.square(vdifs, out=vdifs)
        vdifs *= -1
        np.exp(vdifs, out=vdifs)
        vdifs *= lbda
        vtemp *= vdifs    #broadcasting across color dimension
        uNnewT += vtemp
        uSnewT -= vtemp
        htemp[:] = uET-uWT
        np.fabs(htemp, out=htemp2)
        np.max(htemp2, axis=0, out=hdifs)
        hdifs /= sigma
        np.square(hdifs, out=hdifs)
        hdifs *= -1
        np.exp(hdifs, out=hdifs)
        hdifs *= lbda
        htemp *= hdifs    #broadcasting across color dimension
        uWnewT += htemp
        uEnewT -= htemp
        u[:] = unew
        
#problem 3
#using numexpr and array slices
def perona_malik_color_ne(u, lbda, sigma, iters):
    sinv = np.float32(1./sigma)
    lbda32 = np.float32(lbda)
    unew = u.copy()
    uNT = u[:-1].T
    uST = u[1:].T
    uWT = u[:,:-1].T
    uET = u[:,1:].T
    uNnewT = unew[:-1].T
    uSnewT = unew[1:].T
    uWnewT = unew[:,:-1].T
    uEnewT = unew[:,1:].T
    vdifs = np.empty_like(uNT[0])
    hdifs = np.empty_like(uWT[0])
    temp = np.empty_like(u)
    vtemp = temp[:-1].T
    htemp = temp[:,:-1].T
    temp2 = np.empty_like(temp)
    vtemp2 = temp2[:-1].T
    htemp2 = temp2[:,:-1].T
    for n in xrange(iters):
        ne.evaluate('uST-uNT', out=vtemp)
        ne.evaluate('abs(vtemp)', out=vtemp2)
        np.max(vtemp2, axis=0, out=vdifs)
        ne.evaluate('lbda32*exp(-(sinv*(vdifs))**2)', out=vdifs)
        vtemp *= vdifs #broadcasting
        ne.evaluate('uNnewT+vtemp', out=uNnewT)
        ne.evaluate('uSnewT-vtemp', out=uSnewT)
        ne.evaluate('uET-uWT', out=htemp)
        ne.evaluate('abs(htemp)', out=htemp2)
        np.max(htemp2, axis=0, out=hdifs)
        ne.evaluate('lbda32*exp(-(sinv*(hdifs))**2)', out=hdifs)
        htemp *= hdifs #broadcasting
        ne.evaluate('uWnewT+htemp', out=uWnewT)
        ne.evaluate('uEnewT-htemp', out=uEnewT)
        u[:] = unew
        
        
#vanilla numpy version for problem 4
def min_bias_color(u, lbda, reps):
    #this uses a good bit of excess memory...
    #the approach is based on the approach I took with the
    #optimized versions of problems 2 and 3 below
    difs = np.empty_like(u)
    uN = u[:-1]
    uS = u[1:]
    uW = u[:,:-1]
    uE = u[:,1:]
    uNW = u[:-1,:-1]
    uNE = u[:-1,1:]
    uSW = u[1:,:-1]
    uSE = u[1:,1:]
    Ndifs = difs[:-1]
    Sdifs = difs[1:]
    Wdifs = difs[:,:-1]
    Edifs = difs[:,1:]
    temp = np.empty_like(u)
    #coefs is used to store the coefficients that we use 
    #to control how much diffusion we allow at each pixel
    coefs = np.empty_like(temp[:,:,0])
    vtemp = temp[:-1]
    htemp = temp[:,:-1]
    dtemp = temp[:-1,:-1]
    vdifs = np.empty_like(vtemp[:,:,0])
    hdifs = np.empty_like(htemp[:,:,0])
    ldifs = np.empty_like(dtemp[:,:,0])
    udifs = np.empty_like(dtemp[:,:,0])
    lastrow = difs[-1]
    difsT = difs.T
    coefsT = coefs.T
    for i in xrange(reps):
        #Calculate the difference between the old and new,
        #then add in place.
        vtemp[:] = uN-uS
        np.fabs(vtemp, out=vtemp)
        np.max(vtemp, axis=2, out=vdifs)
        htemp[:] = uW-uE
        np.fabs(htemp, out=htemp)
        np.max(htemp, axis=2, out=hdifs)
        dtemp[:] = uNW-uSE
        np.fabs(dtemp, out=dtemp)
        np.max(dtemp, axis=2, out=ldifs)
        dtemp[:] = uSW-uNE
        np.fabs(dtemp, out=dtemp)
        np.max(dtemp, axis=2, out=udifs)
        makecoefs(vdifs, hdifs, ldifs, udifs, coefs)
        coefs *= lbda
        #overwrite most of difs instead of reinitializing
        Ndifs[:] = uS-uN
        #initialize what's left
        lastrow[:] = 0.
        Sdifs += uN
        Sdifs -= uS
        Wdifs += uE
        Wdifs -= uW
        Edifs += uW
        Edifs -= uE
        difsT *= coefsT#broadcasting
        u += difs
        
        
#numexpr optimized version of problem 4
def min_bias_color_ne(u, lbda, reps):
    #this uses a good bit of excess memory...
    #the approach is based on the approach I took with the
    #optimized versions of problems 2 and 3 below
    difs = np.empty_like(u)
    lb32 = np.float32(lbda)
    uN = u[:-1]
    uS = u[1:]
    uW = u[:,:-1]
    uE = u[:,1:]
    uNW = u[:-1,:-1]
    uNE = u[:-1,1:]
    uSW = u[1:,:-1]
    uSE = u[1:,1:]
    Ndifs = difs[:-1]
    Sdifs = difs[1:]
    Wdifs = difs[:,:-1]
    Edifs = difs[:,1:]
    temp = np.empty_like(u)
    coefs = np.empty_like(temp[:,:,0])
    #coefs is used to store the coefficients that we use 
    #to control how much diffusion we allow at each pixel
    vtemp = temp[:-1]
    htemp = temp[:,:-1]
    dtemp = temp[:-1,:-1]
    vdifs = np.empty_like(vtemp[:,:,0])
    hdifs = np.empty_like(htemp[:,:,0])
    ldifs = np.empty_like(dtemp[:,:,0])
    udifs = np.empty_like(dtemp[:,:,0])
    lastrow = difs[-1]
    difsT = difs.T
    coefsT = coefs.T
    for i in xrange(reps):
        #Calculate the difference between the old and new,
        #then add in place.
        ne.evaluate('abs(uN-uS)', out=vtemp)
        np.max(vtemp, axis=2, out=vdifs)
        ne.evaluate('abs(uW-uE)', out=htemp)
        np.max(htemp, axis=2, out=hdifs)
        ne.evaluate('abs(uNW-uSE)', out=dtemp)
        np.max(dtemp, axis=2, out=ldifs)
        ne.evaluate('abs(uSW-uNE)', out=dtemp)
        np.max(dtemp, axis=2, out=udifs)
        makecoefs(vdifs, hdifs, ldifs, udifs, coefs)
        ne.evaluate('lb32*coefs', out=coefs)
        #overwrite most of difs instead of reinitializing
        ne.evaluate('uS-uN', out=Ndifs)
        #initialize what's left
        lastrow[:] = 0.
        ne.evaluate('Sdifs+uN-uS', out=Sdifs)
        ne.evaluate('Wdifs+uE-uW', out=Wdifs)
        ne.evaluate('Edifs+uW-uE', out=Edifs)
        difsT *= coefsT#broadcasting
        difs *= lb32
        ne.evaluate('u+difs', out=u)