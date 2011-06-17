import scipy as sp

def cdiff(f, xpts, vary=[0], accur=2, degree=1, tol=1e-5):
    if accur not in [2,4,6]:
        raise ValueError("Invalid accuracy.  Must be 2, 4, or 6")
    if degree not in [1,2]:
        raise ValueError("Only degrees 1 and 2 are defined")

    #cdiff = lambda ccfs, hcfs, xval: sum([c*f(x+h*tol) for c,h,x in zip(ccfs, hcfs, [xval]*len(ccfs))])/(tol**float(degree))
    cdiff = lambda co, ho, xvals, i: sp.sum([c*f(*(sp.asarray([x+h*tol if index in i else x for (x,index) in zip(xvals, range(len(xvals)))]))) for c,h in zip(co, ho)], axis=0)/(tol**float(degree))

    if degree==1:
        if accur==2:
            return cdiff([-1.0/2,0,1.0/2], [-1,0,1],xpts,vary)
        if accur==4:
            return cdiff([1.0/12,-2.0/3,0,2.0/3,-1.0/12], [-2,-1,0,1,2],xpts,vary)
        if accur==6:
            return cdiff([-1.0/60,3.0/20,-3.0/4,0,3.0/4,-3.0/20,1.0/60],[-3,-2,-1,0,1,2,3],xpts,vary)
    if degree==2:
        if accur==2:
            return cdiff([1,-2,1],[-1,0,1],xpts,vary)
        if accur==4:
            return cdiff([-1.0/12,4.0/3,-5.0/2,4.0/3,-1.0/12],[-2,-1,0,1,2],xpts,vary)
        if accur==6:
            return cdiff([1.0/90,-3.0/20,3.0/2,-49.0/18,3.0/2,-3.0/20,1.0/90],[-3,-2,-1,0,1,2,3],xpts,vary)

def fbdiff(f, xpts, vary=0, accur=1, degree=1, direction='f',tol=1e-5):
    """Computer forward difference using coefficients from taylor series"""
    if accur not in [1,2,3]:
        raise ValueError("Invalid accuracy.  Must be 1, 2, or 3")
    if degree not in [1,2]:
        raise ValueError("Only degrees 1 and 2 are defined")
    if direction not in ['f', 'b']:
        raise ValueError("Invalid direction.  Must be 'f'orward or 'b'ackward")

    #fdiff = lambda ccfs, hcfs, xval: sum([c*f(x+h*tol) for c,h,x in zip(ccfs, hcfs, [xval]*len(ccfs))])/(tol**float(degree))
    #fdiff = lambda co, ho, xvals: sp.sum([c*f(*(sp.asarray(xvals)+h*tol)) for c,h in zip(co, ho)],axis=0)/(tol**float(degree))
    fdiff = lambda co, ho, xvals, i: sp.sum([c*f(*(sp.asarray([x+h*tol if index in i else x for (x,index) in zip(xvals, range(len(xvals)))]))) for c,h in zip(co, ho)], axis=0)/(tol**float(degree))

    if degree==1:
        if accur==1:
            if direction is 'f':
                return fdiff([-1,1],[0,1],xpts,vary)
            else:
                return fdiff([1,-1],[0,-1],xpts,vary)
        if accur==2:
            if direction is 'f':
                return fdiff([-3.0/2,2,-1.0/2],[0,1,2],xpts,vary)
            else:
                return fdiff([3.0/2,-2,1.0/2],[0,-1,-2],xpts,vary)
        if accur==3:
            if direction is 'f':
                return fdiff([-11.0/6,3,-3.0/2,1.0/3],[0,1,2,3],xpts,vary)
            else:
                return fdiff([11.0/6,-3,3.0/2,-1.0/3],[0,-1,-2,-3],xpts,vary)
    if degree==2:
        if accur==1:
            if direction is 'f':
                return fdiff([1,-2,1],[0,1,2],xpts,vary)
            else:
                return fdiff([-1,2,-1],[0,-1,-2],xpts,vary)
        if accur==2:
            if direction is 'f':
                return fdiff([2,-5,4,-1],[0,1,2,3],xpts,vary)
            else:
                return fdiff([-2,5,-4,1],[0,-1,-2,-3],xpts,vary)
        if accur==3:
            if direction is 'f':
                return fdiff([35.0/12,-26.0/3,19.0/2,-14.0/3,11.0/12],[0,1,2,3,4],xpts,vary)
            else:
                return fdiff([-35.0/12,26.0/3,-19.0/2,14.0/3,-11.0/12],[0,-1,-2,-3,-4],xpts,vary)
