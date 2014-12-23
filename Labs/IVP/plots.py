# IVP Lab
import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from solution import Euler, Midpoint, RK4, harmonic_oscillator_ode
import solution

def Fig1(): 
# Figure #1 in the Lab: The solution of y'=y-2x+4, y(0)=0, is 
# y(x) = -2 + 2x + (ya + 2)e^x. This code plots the solution for 0<x<2,
# and then plots the approximation given by Euler's method
# Text Example
    a, b, ya = 0.0, 2.0, 0.0
    def f(x,ya): 
        return -2. + 2.*x + (ya + 2.)*np.exp(x)
    
    def ode_f(x,y): 
        return np.array([1.*y -2.*x + 4.])
    
    plt.plot(np.linspace(a,b,11), Euler(ode_f,a,b,10,ya) , 'b-',label="h = 0.2")
    plt.plot(np.linspace(a,b,21), Euler(ode_f,a,b,20,ya) , 'g-',label="h = 0.1")
    plt.plot(np.linspace(a,b,41), Euler(ode_f,a,b,40,ya) , 'r-',label="h = 0.05")
    
    x = np.linspace(0,2,200); k =int(200/40)
    plt.plot(x[::k], solution.function(x,f,0.0)[::k], 'k*-',label="Solution") # The solution 
    plt.plot(x[k-1::k], solution.function(x,f,0.0)[k-1::k], 'k-') # The solution 
    
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Fig1.pdf')
    plt.clf()


def Fig2():
# Integral curves for y' = sin y using dopri5 
    a, b, n = 0.0, 5.0, 50
    k, x= n//10, np.linspace(a,b,n+1)
    def ode_f3(x,y): 
        return np.array([np.sin(y)])
    
    def dopri5_integralcurves(ya): 
            test1 = ode(ode_f3).set_integrator('dopri5',atol=1e-7,rtol=1e-8,nsteps=500) 
            y0 = ya; x0 = a; test1.set_initial_value(y0,x0) 
            Y = np.zeros(x.shape); Y[0] = y0
            for j in range(1,len(x)): 
                    test1.integrate(x[j])
                    Y[j]= test1.y
            return Y
    
    plt.plot(x[::k], dopri5_integralcurves(5.0*np.pi/2.0)[::k], 'k-')
    plt.plot(x[::k], dopri5_integralcurves(3.0*np.pi/2.0)[::k], 'k-')
    plt.plot(x[::k], dopri5_integralcurves(7.0*np.pi/4.0)[::k], 'k-')
    plt.plot(x[::k], dopri5_integralcurves(0.0*np.pi/2.0)[::k], 'k-')
    plt.plot(x[::k], dopri5_integralcurves(-np.pi)[::k], 'k*-',label='Equilibrium solutions')
    plt.plot(x[::k], dopri5_integralcurves(np.pi)[::k], 'k*-')
    plt.plot(x[::k], dopri5_integralcurves(2*np.pi)[::k], 'k*-')
    plt.plot(x[::k], dopri5_integralcurves(3*np.pi)[::k], 'k*-')
    plt.plot(x[::k], dopri5_integralcurves(np.pi/4.0)[::k], 'k-')
    plt.plot(x[::k], dopri5_integralcurves(np.pi/2.0)[::k], 'k-')
    plt.plot(x[::k], dopri5_integralcurves(-np.pi/2.0)[::k], 'k-')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Fig2.pdf')
    plt.clf()


def Fig3():
    a, b, ya = 0., 2., 0.
    def f(x,ya):
        return -2. + 2.*x + 2.*np.exp(x)
    
    def ode_f(x,y): 
        return np.array([1.*y -2.*x + 4.])
    
    N = np.array([10,20,40,80,160])  # Number of subintervals
    Euler_sol, Mid_sol, RK4_sol = np.zeros(len(N)), np.zeros(len(N)), np.zeros(len(N))
    for j in range(len(N)):
            Euler_sol[j] = Euler(ode_f,a,b,N[j],ya)[-1]
            Mid_sol[j] = Midpoint(ode_f,a,b,N[j],ya)[-1]
            RK4_sol[j] = RK4(ode_f,a,b,N[j],ya,dim=1)[-1][0]
    
    print Euler_sol, Mid_sol, RK4_sol
    print "Answer = ", f(2.,0.)
    h = 2./N
    plt.loglog(h, abs(( Euler_sol - f(2.,0.))/f(2.,0.) ), '-b', label="Euler method"   , linewidth=2.)
    plt.loglog(h, abs(( Mid_sol - f(2.,0.))/f(2.,0.) ),   '-g', label="Midpoint method", linewidth=2. )
    plt.loglog(h, abs(( RK4_sol - f(2.,0.))/f(2.,0.) ),   '-k', label="Runge Kutta 4"  , linewidth=2. )
    plt.xlabel("$h$", fontsize=16)
    plt.ylabel("Relative Error", fontsize = 16)
    # plt.title("loglog plot of relative error in approximation of $y(2)$.")
    plt.legend(loc='best')
    plt.savefig('Fig3.pdf')
    # plt.show()
    plt.clf()


def Exercise1(): 
    a, b, ya = 0.0, 2.0, 0.0
    def f(x,ya): 
        return 4. - 2.*x + (ya - 4.)*np.exp(-x)
    
    def ode_f(x,y): 
        return np.array([-1.*y -2.*x + 2.])
    
    
    plt.plot(np.linspace(a,b,11), Euler(ode_f,a,b,10,ya) , 'b-',label="h = 0.2")
    plt.plot(np.linspace(a,b,21), Euler(ode_f,a,b,20,ya) , 'g-',label="h = 0.1")
    plt.plot(np.linspace(a,b,41), Euler(ode_f,a,b,40,ya) , 'r-',label="h = 0.05")
    
    x = np.linspace(0,2,200); k =int(200/40)
    plt.plot(x[::k], solution.function(x,f,0.0)[::k], 'k*-',label="Solution") # The solution 
    plt.plot(x[k-1::k], solution.function(x,f,0.0)[k-1::k], 'k-') # The solution 
    
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Exercise1.pdf')
    # plt.show()
    plt.clf()


def Exercise2():
    # Euler's method corresponds to the left hand sum.
    # Backward Euler's method corresponds to the right hand sum.
    # Modified Euler's method corresponds to the trapezoidal rule.
    # The midpoint method corresponds to the midpoint rule.
    # RK4 corresponds to Simpson's rule.
    return


def Exercise3():
    a, b, ya = 0., 2., 0.
    def f(x,ya): 
        return 4. - 2.*x + (ya - 4.)*np.exp(-x)
    
    def ode_f(x,y): 
        return np.array([-1.*y -2.*x + 2.])
    
    N = np.array([10,20,40,80,160])  # Number of subintervals
    Euler_sol, Mid_sol, RK4_sol = np.zeros(len(N)), np.zeros(len(N)), np.zeros(len(N))
    for j in range(len(N)):
            Euler_sol[j] = Euler(ode_f,a,b,N[j],ya)[-1]
            Mid_sol[j] = Midpoint(ode_f,a,b,N[j],ya)[-1]
            RK4_sol[j] = RK4(ode_f,a,b,N[j],ya,dim=1)[-1][0]
    
    print Euler_sol, Mid_sol, RK4_sol
    print "Answer = ", f(2.,0.)
    h = 2./N
    plt.loglog(h, abs(( Euler_sol - f(2.,0.))/f(2.,0.) ), '-b', label="Euler method"   , linewidth=2.)
    plt.loglog(h, abs(( Mid_sol - f(2.,0.))/f(2.,0.) ),   '-g', label="Midpoint method", linewidth=2. )
    plt.loglog(h, abs(( RK4_sol - f(2.,0.))/f(2.,0.) ),   '-k', label="Runge Kutta 4"  , linewidth=2. )
    plt.xlabel("$h$", fontsize=16)
    plt.ylabel("Relative Error", fontsize = 16)
    plt.title("loglog plot of relative error in approximation of $y(2)$.")
    plt.legend(loc='best')
    plt.savefig("Exercise3.pdf")
    # plt.show()
    plt.clf()
    

def Exercise4old(): 
# Integral curves for f(x).
    a , b, n = 0.0,  1.6,  200
    k, x = int(n/20), np.linspace(a,b,n+1)
    def f(x,ya): 
        return 4. - 2.*x + (ya - 4.)*np.exp(-x)
    
    plt.plot(x, solution.function(x,f,0.0), 'k-')
    plt.plot(x, solution.function(x,f,2.0), 'k-')
    plt.plot(x[::k], solution.function(x,f,4.0)[::k], 'k*-', label='Particular solution for 'r"$y' +y =  - 2x + 2 $.")
    plt.plot(x, solution.function(x,f,6.0), 'k-')
    plt.plot(x, solution.function(x,f,8.0), 'k-')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
    plt.savefig('Exercise4.pdf')
    plt.clf()


def HOFig1(): 
    # Example comparing damped with nondamped harmonic oscillator
    a, b, ya = 0.0, 50.0, np.array([2., 0.])		# Parameters
    
    m , gamma, k, F = 1, .125, 1,lambda x: 0
    func = lambda x,y: solution.harmonic_oscillator_ode(x,y,m,gamma,k,F)
    Y = solution.RK4(func,a,b,600,ya,dim=2) # 2 dimensional system
    plt.plot(np.linspace(a,b,601), Y[:,0], 'k',linestyle='-')
    
    m , gamma, k, F = 1., 0., 1.,lambda x: 0.
    func = lambda x,y: solution.harmonic_oscillator_ode(x,y,m,gamma,k,F)
    Y = solution.RK4(func,a,b,600,ya,2)
    plt.plot(np.linspace(a,b,601), Y[:,0], 'k-',linestyle='--')
    
    plt.axhline(color='k',linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('HOFig1.pdf')
    # plt.show()
    plt.clf()


def HOFig2(): 
    # Computes relative error for nondamped oscillator
    a, b, n, ya = 0.0, 20.0, 100,np.array([2., 0.])		# Parameters
    
    m , gamma, k, F = 1., 0., 1.,lambda x: 0.
    func = lambda x,y: solution.harmonic_oscillator_ode(x,y,m,gamma,k,F)
    Y = solution.RK4(func,a,b,n,ya,2)
    plt.plot(np.linspace(a,b,n+1), Y[:,0], 'k-')#,linestyle='--')
    ###################################################
    #	Computing relative error of approximation     #
    Y_coarse = solution.RK4(func,a,b,n/2,ya,2)
    
    Relative_Error = np.abs(Y_coarse[-1,0] - Y[-1,0])/np.abs(Y[-1,0])
    print "Relative Error = ", Relative_Error
    ###################################################
    
    plt.axhline(color='k',linestyle='-')
    plt.axvline(color='k',linestyle='-')
# 	plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('HOFig2.pdf')
    # plt.show()
    plt.clf()
    

def Exercise4(): 
# Parameters
    a, b, ya = 0.0, 20.0, np.array([2., -1.])
    
    m , gamma, k, F = 3., 0., 1.,lambda x: 0.
    func1 = lambda x,y: solution.harmonic_oscillator_ode(x,y,m,gamma,k,F)
    Y1 = solution.RK4(func1,a,b,800,ya,2) # 2 dimensional system
    plt.plot(np.linspace(a,b,801), Y1[:,0], 'k',linestyle='-')
    
    m , gamma, k, F = 1, 0, 1,lambda x: 0
    func2 = lambda x,y: solution.harmonic_oscillator_ode(x,y,m,gamma,k,F)
    Y2 = solution.RK4(func2,a,b,800,ya,2)
    plt.plot(np.linspace(a,b,801), Y2[:,0], 'k-',linestyle='--')
    
    ###################################################
    #	Computing relative error of approximation     #
    m , gamma, k, F = 3., 0., 1.,lambda x: 0.
    # Need about 70 subintervals to get Relative error< 5*10^{-5}
    Y_coarse = solution.RK4(func1,a,b,70,ya,2)
    
    Relative_Error = np.abs(Y_coarse[-1,0] - Y1[-1,0])/np.abs(Y1[-1,0])
    print "Relative Error = ", Relative_Error
    ###################################################
    plt.axhline(color='k',linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Exercise4.pdf')
    # plt.show()
    plt.clf()
    

def Exercise5(): 
    a, b, ya = 0.0, 20.0, np.array([1., -1.])		# Parameters
    # Needs about 180 subintervals to achieve Rel Error < 5*10**(-5)
    
# Damped Oscillators
    def plot_function(param,color):
            func = lambda x,y: harmonic_oscillator_ode(x,y,m=1.,gamma=param,k=1.,F=lambda x: 0.)
            Y = RK4(func,a,b,800,ya,dim=2) 
            plt.plot(np.linspace(a,b,801), Y[:,0], color,linestyle='-',linewidth=2.0)
            Relative_Error = np.abs( Y[-1,0] - RK4(func,a,b,180,ya,2)[-1,0] )/np.abs(Y[-1,0])
            print "Relative Error = ", Relative_Error
            return
    
    plot_function(.5,'k')
    plot_function(1.,'b')
    
    plt.axhline(color='k',linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Exercise5.pdf')
    # plt.show()
    plt.clf()
    

def Exercise6(): 
# Parameters: Interval = [a,b], n = number of subintervals, ya = y(a) 
    a, b, n, ya = 0.0, 40.0, 600, np.array([2., -1.])
    m, k = 2., 2. 
    
# A Forced Oscillator with Damping: m*y'' + gamma*y' + k*y = F(x)
# Requires about 300 subintervals for Rel Error < 5*10**(-5)
    def print_func(gamma, omega,color):
            func = lambda x,y: harmonic_oscillator_ode(x,y,m,gamma,k,lambda x: 2.*np.cos(omega*x))
            Y = RK4(func,a,b,n,ya,dim=2) 
            plt.plot(np.linspace(a,b,n+1), Y[:,0], color,linestyle='-',linewidth=2.0)
            Relative_Error = np.abs(Y[-1,0] - RK4(func,a,b,n/2,ya,2)[-1,0])/np.abs(Y[-1,0])
            print "Relative Error = ", Relative_Error
            return
    
    print_func(.5,1.5,'k')
    print_func(.1,1.1,'b')
    print_func(0.,1.,'g')
    
    plt.axhline(color='k',linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Exercise6.pdf')
    # plt.show()
    plt.clf()
    

if __name__ == "__main__":
    Fig1()
    Fig2()
    Fig3()
    Exercise1()
    Exercise2()
    Exercise3()
    Exercise4()
    Exercise5()
    Exercise6()

# HOFig1()
# HOFig2()
