
from math import sqrt


def quadForm(a,b,c):
    descr = sqrt(b**2-4*a*c)
    x1 = (-b+descr)/2.0*a
    x2 = (-b-descr)/2.0*a
    return (x1,x2)
    
def quadForm2(a,b,c):
    descr = sqrt(b**2-4*a*c)
    sigb = -1.0 if a<0 else 1.0 if a>0 else 0.0
    x1 = (-b-sigb*descr)/2.0*a
    x2 = c/(a*x1)
    return (x1,x2)

def quadFunc2(a,b,c):
    descr = sqrt(b**2-4*a*c)
    x1 = (-b-sig(b)*descr)/2.0*a
    x2 = c/(a*x1)
    return (x1, x2)
    
def quadSFunc2(a,b,c):
    descr = sqrt(b**2-4*a*c)
    def sign(a):
        return -1.0 if a<0 else 1.0 if a>0 else 0.0
        
    x1 = (-b-sign(b)*descr)/2.0*a
    x2 = c/(a*x1)
    return (x1, x2)
    
def sig(a):
    return -1.0 if a<0 else 1.0 if a>0 else 0.0