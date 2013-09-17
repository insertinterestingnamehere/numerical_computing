import scipy as sp
from matplotlib import pyplot
#from scipy.optimize import newton

def newton(func, dfunc, x0, iter=20):
    #we define our own newton function because the scipy function
    #can't handle complex numbers
    x = x0
    for j in xrange(iter):
        x = x-func(x)/dfunc(x)
    return x

#newtonv = sp.vectorize(newton)

def real_basins(func, interval, npoints):
    """Plot func over the real interval and the
    basins of attraction with npoints resolution"""

    x = sp.linspace(interval[0], interval[1], npoints)
    y = sp.array(map(func, x))

    #import available colors in
    colors = pyplot.cm.colors.cnames.keys()

    #remove black since our function will be plotted in black
    colors.remove('black')

    #Use Newton's method to find the roots of func
    #returns an array with length x
    basins = newtonv(func, x)

    #set the color for each seed value
    tol = 1e-7
    valcol = []
    for val in range(len(basins)):
        if val == 0:
            valcol.append(colors[0])
            continue
        if not(abs(basins[val]-basins[val-1]) <= tol):
            valcol.append(colors.pop(sp.random.randint(1,len(colors))))
        else:
            valcol.append(valcol[-1])

    #print valcol

    #plot our function
    pyplot.plot(x,y, 'k')

    #loop through our array and find the associated color for each root of our function
    for col in range(len(valcol)):
        count = valcol.count(col)
        pyplot.plot(x[col], 0, marker=',', color=valcol[col])

    pyplot.show()

def complex_basins(func, dfunc, coeffs, interval, npoints):
    """Plot func over the complex interval and the
    basins of attraction with npoints resolution"""

    seeds = sp.zeros((2, npoints, npoints), dtype='complex128')
    seeds[0:,:,] = sp.ones((npoints, 1))*sp.linspace(interval[0], interval[1], npoints)
    seeds[0:,:,] = seeds[0:,:,] + 1j*seeds[0:,:,]

    colors = pyplot.cm.colors.cnames.keys()
    colors.remove('black')
    clen = len(colors)
    for i in xrange(npoints):
        for j in xrange(npoints):
            seeds[1,i,j] = newton(func, dfunc, seeds[0,i,j])

    #return seeds
    #find the unique roots of seed values
    roots, indices = sp.unique(seeds[1])
    #for each root, associate a color
    col = sp.zeros_like(roots.shape)

    for i in range(len(roots)):
        col[i] = colors.pop(sp.random.randint(clen))



    #seeds[1] = newtonv(func, )

    #set the color for each
    #tol = 1e-7
    #valcol = [[]]
    #for valx in range(npoints):
        #for valy in range(npoints):
            #if valx and valy == 0:
                #valcol.append(colors[0])
                #continue
            #if not(abs(seeds[1,valx,valy]-seeds[1,valx,valy-1]) <= tol):
                #valcol.append(colors.pop(sp.random.randint(1,len(colors))))
            #else:
                #valcol.append(valcol[0][0])

    #for col in range(npoints):
        #for col1 in range(npoints):
            #pyplot.plot(seeds[0,col,col1], seeds[1,col,col1], marker=',', color=valcol[col][col1])

    tol = 1e-07
    roots = sp.roots(coeffs)
    pyplot.hold(True)
    #for i in range(len(roots)):
        #pyplot.plot(seeds[abs(seeds[1:,:,]-roots[i])<tol], color=colors.pop(sp.random.randint(len(colors))), linestyle='None', marker='.')
    pyplot.pcolor(seeds[0], seeds[1])
    pyplot.hold(False)
    pyplot.show()
