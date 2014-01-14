# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from cvxopt import matrix ,solvers
import numpy as np

# <codecell>

def Problem1():
    c = matrix([2., 1., 3.])
    G= matrix([[-1., -2., -1., 0.,0.],[-2., -1., 0., -1.,0.],[0., -3., 0., 0.,-1.]])
    h = matrix([ -3., -10., 0., 0.,0.])
    sol = solvers.lp(c,G,h)
    return sol['x'],sol['primal objective']

# <codecell>

#x,y=Problem3()
#print x
#print y

# <codecell>

def Problem2():
    c = matrix([4., 7., 6., 8., 8., 9])
    G= matrix([[-1., 0., 0., -1., 0., -1., 0., 0., 0., 0., 0.],[-1., 0., 0., 0., -1., 0., -1., 0., 0., 0., 0.], [0., -1., 0., -1., 0., 0., 0., -1., 0., 0., 0.], [0., -1., 0., 0., -1., 0., 0., 0., -1., 0., 0.], [0., 0., -1., -1., 0., 0., 0., 0., 0., -1., 0.],[0., 0., -1., 0., -1., 0., 0., 0., 0., 0., -1.]])
    h = matrix([-7., -2., -4., -5., -8., 0., 0., 0., 0., 0., 0.])
    sol = solvers.lp(c,G,h)  
    return sol['x'],sol['primal objective']

# <codecell>

def Problem3():
    Q= matrix([[3., 2., 1.],[2., 4., 2.],[1., 2., 3. ]])
    p=matrix([3., 0., 1.])
    sol=solvers .qp(Q, p)
    return sol['x'],sol['primal objective']

# <codecell>

def Problem4():
    datam=np.load('ForestData.npy')
    c=matrix(datam[:,3]*-1)
    G=np.zeros((21,7+3+21))
    h=np.zeros(7+3+21)
    G[:,-21:]=-1*np.eye(21)
    h[:7]=datam[::3,1]
    for i in xrange(7):
        G[i*3:(i+1)*3,i]=np.ones(3)
    G[:,7]=-1*datam[:,4]
    G[:,8]=-1*datam[:,5]
    G[:,9]=-1*datam[:,6]
    h[7]=-40000
    h[8]=-5
    h[9]=-70*788
    G=G.T
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    sol = solvers.lp(c,G,h)  
    return sol['x'],sol['primal objective']*-1000

# <codecell>

#x,y=Problem4()
#print x
#print y

# <codecell>

'''
forest=np.array([[1,75.,1,503.,310.,0.01,40],
[0,0,2,140,50,0.04,80],
[0,0,3,203,0,0,95],
[2,90.,1,675,198,0.03,55],
[0,0,2,100,46,0.06,60],
[0,0,3,45,0,0,65],
[3,140.,1,630,210,0.04,45],
[0,0,2,105,57,0.07,55],
[0,0,3,40,0,0,60],
[4,60.,1,330,112,0.01,30],
[0,0,2,40,30,0.02,35],
[0,0,3,295,0,0,90],
[5,212.,1,105,40,0.05,60],
[0,0,2,460,32,0.08,60],
[0,0,3,120,0,0,70],
[6,98.,1,490,105,0.02,35],
[0,0,2,55,25,0.03,50],
[0,0,3,180,0,0,75],
[7,113.,1,705,213,0.02,40],
[0,0,2,60,40,0.04,45],
[0,0,3,400,0,0,95]])
'''
#np.save('ForestData',forest)

# <codecell>


