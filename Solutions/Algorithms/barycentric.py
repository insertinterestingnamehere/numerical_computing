import scipy as sp


def Lj(ip_x, xpts):
        """Build array of Lagrange basis functions for ip_x

        Returns a 2D array of basis vectors as rows"""
        n = len(ip_x)
        L =[]
        for j in range(n):
            a = 1
            for k in range(n):
                if j!=k:
                    a *= (xpts-ip_x[k])/(ip_x[j]-ip_x[k])
            L.append(a)
        return sp.vstack(L)

def lagrange2(ip_x, ip_y, interval, npoints=500):
    """Faster lagrange calculator."""
    interval = sp.linspace(interval[0],interval[1],float(npoints))

    #calculate barycentric weights
    c = (max(ip_x)-min(ip_x))/4.0
    w = sp.zeros_like(ip_x.T)
    shuffle = sp.random.permutation(len(ip_x)-1)
    for k in range(len(ip_x)):
        test = (ip_x[k]-ip_x)/c
        test = sp.delete(test, k)
        test = test[shuffle]
        w[k] = 1.0/sp.prod(test)

    #calculate p(x)
    numer = sp.sum([(w[k]*ip_y[k])/(interval-ip_x[k]) for k in range(len(ip_x))], axis=0)
    denom = sp.sum([w[k]/(interval-ip_x[k]) for k in range(len(ip_x))], axis=0)
    return numer/denom

def lagrange1(ip_x, ip_y, interval, npoints=500):
    """Calculate Lagrange Interpolant

    INPUTS:
        ip_x    X input values (range)
        ip_y    Function evalated at each X value
        interval    Interval to interpolate
        npoints     Number of interpolation points to use

    RETURN:
        result  Vector containing coefficients of Lagrange polynomial
    """
    interval = sp.linspace(interval[0], interval[1], float(npoints))
    L = Lj(ip_x, interval)
    #sum the entries together
    result = sp.sum([ip_y[pt]*L[pt] for pt in range(len(L))], axis=0)
    return result

def cheb_nodes(n):
    """Calculate ip_x"""
    i = sp.arange(1.0,n)
    return sp.cos(((2.0*i-1)*sp.pi)/(2.0*n))
