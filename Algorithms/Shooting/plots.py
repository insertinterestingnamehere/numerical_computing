from __future__ import division

import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt


def Figure_Cannon_with_AirResistance():
    def Cannon(b= 150,ya=0.,va=35.,phi=np.pi/4,nu=0.):
        g = 9.8067
        def ode_f(x,y): 
            # y = [z,v,phi]
            return np.array([np.tan(y[2]), -(g*np.sin(y[2]) + nu*y[1]**2.)/(y[1]*np.cos(y[2])), 
                    -g/y[1]**2.])
    
        
        a= 0.
        abstol,reltol= 1e-4,1e-4
        
        dim, T = 3, np.linspace(a,b,801)
        example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
        example.set_initial_value(np.array([ya,va,phi]),a)
        Y = np.zeros((len(T),dim))
        Y[0,:] = np.array([0.,.5,np.pi/4.])
        for j in range(1,len(T)): 
            Y[j,:] = example.integrate(T[j])
            if Y[j,0]<(-1e-3): break
            
        return T, T[:j],Y[:j,0]
    
    
    T,X,Y = Cannon(nu = 0.,va = 45,phi=np.pi/3,b=200)
    T,X1,Y1 = Cannon(nu = 0.0003,va = 45,phi=np.pi/3,b=200)
    #plt.plot(T,np.zeros(T.shape),'-k',np.zeros(10),np.linspace(0,80,10),'-k')
    plt.plot(X1, Y1, '--r', linewidth=2.0)
    plt.plot(X1[-1], Y1[-1], 'or', markersize=6.)
    plt.plot(X, Y, '-k', linewidth=2.0)
    plt.plot(X[-1], Y[-1], 'ok', markersize=6.)
    
    plt.axis([0, 200, 0, 85])
    plt.savefig("Cannon_with_AirResistance.pdf")
    plt.clf()
    

def Exercise1():
    def find_t(f,a,b,alpha,beta,t0,t1,maxI):
        
        sol1 = 0
        i = 0
        print "Guesses = ", t0,t1
        while( abs(sol1-beta) > 10**-8 and i < maxI):
            sol0 = odeint(f,np.array([alpha,t0]), [a,b],atol=1e-10)[1,0]
            sol1 = odeint(f,np.array([alpha,t1]), [a,b],atol=1e-10)[1,0]
            t2 =  t1 - ( sol1 - beta)*(t1-t0)/( sol1-sol0)
            t0 = t1
            t1 = t2
            i = i+1
        if(i is maxI):
            print "t not found"
        print "Iterations = ", i
        return t2

    
    def solveSecant(f,X,a,b,alpha,beta,t0,t1,maxI):
        t = find_t(f,a,b,alpha,beta,t0,t1,maxI)
        print "Final t = ", t
        sol = odeint(f,np.array([alpha,t]), X,atol=1e-10)[:,0]
            
        return sol
    
    
    def ode(y,x): 
        return np.array([y[1], -4*y[0]-9*np.sin(x)])
    
    
    alpha, beta = 1,1
    X = np.linspace(0,np.pi,100)
    Y1 = solveSecant(ode,X,0,np.pi,alpha,beta,3.,1.,40)
    Y2 = solveSecant(ode,X,0,np.pi,alpha,beta,-5,-6,40)
    
    plt.plot(X,Y1,'-b',linewidth=1.5)
    plt.plot(X,Y2,'-g',linewidth=1.5)
    plt.savefig('Fig1.pdf')
    plt.clf()
    

def Exercise2():
    # y''(x) = 3 + 2*y/x^2, x  in [1,e], y(1) = 6, y(e)= e^2 + 6/e
    # Exact Solution: y(x) = x^2*ln(x)+6*x^(-1)
    # y'(x) = 2xln(x) + x^2*(1/x) - 6x^(-2)
    # y'(x) = 2xln(x) + x - 6x^(-2)
    # y''(x)= 2 ln(x)+2+1 +12x^(-3)
    # y''(x) = 3 + 2*y/x^2
    
    def find_t(f,a,b,alpha,beta,t0,maxI):
        y = 0
        i = 0
        print "Guess: t =", t0
        while( abs(y-beta) > 10**-8 and i < maxI):
            sol = odeint(f,np.array([alpha,t0,0,1]), [a,b],atol=1e-10)
            y, z  = sol[1,0], sol[1,2]
            t1 =  t0 - ( y - beta)/z
            t0 = t1
            i = i+1
        if(i is maxI):
                print "t not found"
        print "Iterations = ", i
        return t1

    
    def solveSecant(f,X,a,b,alpha,beta,t0,maxI):
        t = find_t(f,a,b,alpha,beta,t0,maxI)
        print "Final t = ", t
        sol = odeint(f,np.array([alpha,t,0,1]), X,atol=1e-10)[:,0]
        return sol
    
    
    def ode(y,x): 
        return np.array([y[1] , 3+2.*y[0]/x**2.,y[3],
                (2./x**2.)*y[2] ])
    
    a,b = 1.,np.exp(1.)
    alpha, beta = 6.,np.exp(2.) + 6.*np.exp(-1.)
    
    X = np.linspace(a,b,100)
    Y1 = solveSecant(ode,X,a,b,alpha,beta,150,40)
    
    plt.plot(X,Y1,'-k',linewidth=1.5)
    # Y2 = (X**2.)*np.log(X)+6*(X**(-1.))		# Exact solution
    # plt.plot(X[::5],Y2[::5],'*r')
    plt.xlabel('$x$',fontsize=16)
    plt.ylabel('$y$',fontsize=16)
    plt.savefig('Fig2.pdf')
    plt.clf()
    

def Exercise3():
    def find_t(f,a,b,za,va,beta,t0,t1,maxI):
        sol1 = -1
        i = 0
        print "Guesses = ", t0,t1
        while( abs(sol1-beta) > 10**-8 and i < maxI):
            sol0 = odeint(f,np.array([za,va,t0]), [a,b],atol=1e-10)[1,0]
            sol1 = odeint(f,np.array([za,va,t1]), [a,b],atol=1e-10)[1,0]
            t2 =  t1 - ( sol1 - beta)*(t1-t0)/( sol1-sol0)
            t0 = t1
            t1 = t2
            i = i+1
        if(i is maxI):
            print "t not found"
        print "Iterations = ", i
        return t2

    
    def solveSecant(f,X,a,b,za,va,beta,t0,t1,maxI):
        t = find_t(f,a,b,za,va,beta,t0,t1,maxI)
        sol = odeint(f,np.array([za,va,t]), X,atol=1e-10)[:,0]
        return sol
    
    g = 9.8067
    def ode(y,x,nu=.0003): 
        # y = [z,v,phi]
        return np.array([np.tan(y[2]), -(g*np.sin(y[2]) + nu*y[1]**2.)/(y[1]*np.cos(y[2])), 
        -g/y[1]**2.])
    
    a, b = 0., 195
    za, va = 0., 45 # Initial_Conditions = np.array([z=za,v=va,phi=t])
    beta =  0.
    X = np.linspace(a,b,100)
    Y1 = solveSecant(ode,X,a,b,za,va,beta,np.pi/4., np.pi/4.5,40)
    Y2 = solveSecant(ode,X,a,b,za,va,beta,np.pi/3.5, np.pi/3.0,40)
    
    new_ode = lambda y,x,nu=0.: ode(y,x,nu)
    Y3 = solveSecant(new_ode,X,a,b,za,va,beta,np.pi/4., np.pi/4.5,40)
    Y4 = solveSecant(new_ode,X,a,b,za,va,beta,np.pi/3.5, np.pi/3.0,40)
    
    plt.plot(X,Y1,'-b',linewidth=1.5,label=r"$\nu = 0.0003$")
    plt.plot(X,Y2,'-b',linewidth=1.5)
    plt.plot(X,Y3,'-k',linewidth=1.5,label=r"$\nu = 0.0$")
    plt.plot(X,Y4,'-k',linewidth=1.5)
    plt.legend(loc='best')
    plt.xlabel('$x$',fontsize=16)
    plt.ylabel('$y$',fontsize=16)
    plt.savefig('Cannon_Shooting.pdf')
    plt.clf()
    

if __name__ == "__main__":
    Figure_Cannon_with_AirResistance()
    Exercise1()
    Exercise2()
    Exercise3()

