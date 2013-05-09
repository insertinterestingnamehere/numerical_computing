import scipy as sp
from mytimer import timer
from matplotlib import pyplot as plt
from scipy import linalg as la

i = arange(1500, 2500+200, 200)

with timer(loops=1, repeats=1) as t:
    for n in i:
        print "Array size: %d x %d" % (n ,n)
        t.time(sp.dot, sp.rand(n,n), sp.rand(n,n))

y = [z[0] for z in t.results]
X = sp.column_stack([sp.log(i), sp.ones_like(i)])
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()
