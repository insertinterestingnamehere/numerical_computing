from math import sqrt
import scipy as sp

def Problem1_1(*args, **kargs):
    return 42
def Problem1_2(a,b):
    return a*b
def Problem1_3():
    print "You called!"

def quadForm(a,b,c):
    descr = sqrt(b**2-4*a*c)
    x1 = (-b+descr)/2.0*a
    x2 = (-b-descr)/2.0*a
    return (x1, x2)
    
def quadForm2(a,b,c):
    descr = sqrt(b**2-4*a*c)
    sigb = b*(-1.0 if b<0 else 1.0 if b>0 else 0.0)
    x1 = (-b*sigb*descr)/2.0*a
    x2 = c/(x1*a)
    return (x1, x2)

def Problem3_1(x):
    return (x**2>10) or (x>0 and x<2)
def Problem3_2(x):
    from scipy.special import yn
    return (abs(sp.absp.yn(1,x))>1)

def Problem4(x):
    x = sp.asarray(x)
    mi, ma = x.min(), x.max()
    rand = sp.random.uniform(mi, ma, size=x.shape)

    y = sp.zeros_like(x)
    y[range(0,x.size,2)]=x[x>rand]
    y[range(1,x.size-1,2)]=x[x<=rand]
    return y
    