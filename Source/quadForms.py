from math import sqrt

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