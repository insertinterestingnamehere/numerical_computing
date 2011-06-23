import scipy as sp
from timer import timer
from matplotlib import pyplot as plt
from scipy.linalg import lstsq
from scipy import dot, arange

i = arange(1500, 2500+200, 200)
k = 1

def multArray(A, B):
    return dot(A,B)    

y = []
with timer(loops=1, repeats=1) as t:
    for n in i:
        t.time(multArray, sp.rand(n,n), sp.rand(n,n))
        y.append(t.results[0][0])

X = sp.column_stack([sp.log(i), sp.ones_like(i)])
sol = lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()