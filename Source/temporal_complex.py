import scipy as sp
from timer import timer
from matplotlib import pyplot as plt
from scipy.linalg import lstsq

i = sp.arange(1500, 2500+200, 200)
k = 1

def multArray(A, B):
    return sp.dot(A,B)    

y = []
for n in i:
    with timer(loops=20) as t:
        a = t.time(multArray, sp.rand(n,n), sp.rand(n,n))
        y.append(a[0])

X = sp.row_stack([sp.log(i), sp.ones_like(i)])
sol = lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()