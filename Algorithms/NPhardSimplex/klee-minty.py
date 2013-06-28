from cvxopt import matrix, solvers, glpk

import numpy as np
'''

c = []
h = []
G = []

n = 10
for i in range(n):
    c.append(-2.**(n-1-i))
##print c


for i in range(1,n+1):
    h.append(1.**(i))
for i in range(n):
    h.append(0.)
##print h

for i in range(n):
    row = []
    for k in range(i):
        row.append(0.)
    row.append(1.)
    if i != n-1:
        row.append(4.)
        for j in range(3,n+1-i):
            row.append(2.**(j))
    for j in range(n):
        if i == j:
            row.append(-1.)
        else:
            row.append(0.)
##    print row
    G.append(row)
##print G

c = np.array(c)
h = np.array(h)
G = np.array(G)

c = matrix(c)
h = matrix(h)
G = matrix(G.T)
'''



n = 100
epsilon = round(1./3,3)

c = []
G = []
h = []
for i in range(n-1):
    c.append(0.)
c.append(1.)

for i in range(n):
    row = []
    for j in range(2*i):
        row.append(0.)
    row.append(1.)
    row.append(-1.)
    if i != n-1:
        row.append(epsilon)
        row.append(epsilon)
        if i != n-2:
            for j in range((2*n-4-2*i)):
                row.append(0.)
    G.append(row)

for i in range(n):
    h.append(1.)
    h.append(0.)

##print G


c = np.array(c)
G = np.array(G)
h = np.array(h)

##print c
##print G.T
print G.shape
##print h
c = matrix(c)
G = matrix(G.T)
h = matrix(h)

##print G
sol = solvers.lp(c, G, h)
print sol
print sol['x']
##sol = glpk.ilp(c,G,h)#,I=set([0,1]))
##print sol[1]
