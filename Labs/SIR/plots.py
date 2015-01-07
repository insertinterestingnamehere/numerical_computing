from __future__ import division

import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from scipy.integrate import ode
import numpy as np
from math import pi, cos
import matplotlib.pyplot as plt
from solution import SIR
from scikits import bvp_solver


def Example():
    a, ya, b = 0., 2., 1.6

    def ode_f(t, y):
        return np.array([-1. * y + 6. + 2. * t])

    ode_object = ode(ode_f).set_integrator('dopri5', atol=1e-5)
    ode_object.set_initial_value(ya, a)

    dim, t = 1, np.linspace(a, b, 51)
    Y = np.zeros((len(t), dim))
    Y[0, :] = ya
    for j in range(1, len(t)):
        Y[j, :] = ode_object.integrate(t[j])

    plt.plot(t, Y[:, 0], '-k')
    plt.axis([a, b, ya, 8])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('Example1.pdf')
    plt.clf()
    return t, Y.T[0]


def Exercise1():
    a, b, ya = 0., 16., np.array([0, 1, -2])

    def ode_f(t, y):
        return np.array([y[1], y[2], -.2 * (y[1] + 2. * y[0])])

    example = ode(ode_f).set_integrator('dopri5', atol=1e-8)
    example.set_initial_value(ya, a)

    dim, t = 3, np.linspace(a, b, 201)
    Y = np.zeros((len(t), dim))
    Y[0, :] = ya
    for j in range(1, len(t)):
        Y[j, :] = example.integrate(t[j])

    plt.plot(t, Y[:, 0], '-k')
    plt.axis([a - 1., b + 1, -200, 400])
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig("exercise1.pdf")
    plt.clf()


def Exercise2():  # SIR
    beta, gamma = 0.5, 0.25   # Exercise 2
    a, b, ya = 0., 100., np.array([1. - (6.25e-7), 6.25e-7, 0.])

    t, Y = SIR(a, b, beta, gamma, ya)
    print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:, 1])

    plt.plot(t, Y[:, 0], '-k', label='Susceptible')
    plt.plot(t, Y[:, 2], '-b', label='Recovered')
    plt.plot(t, Y[:, 1], '-r', label='Infected')
    plt.axis([a, b, -.1, 1.1])
    plt.legend(loc=1)
    plt.xlabel('T (days)')
    plt.ylabel('Proportion of Population')
    plt.savefig("SIR1.pdf")
    plt.clf()


def Exercise2a():  # SIR
    beta, gamma = 1., 1. / 3.   # Exercise 3a
    a, b, ya = 0., 50., np.array([1. - (1.667e-6), 1.667e-6, 0.])

    t, Y = SIR(a, b, beta, gamma, ya)
    print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:, 1])

    plt.plot(t, Y[:, 0], '-k', label='Susceptible')
    plt.plot(t, Y[:, 2], '-b', label='Recovered')
    plt.plot(t, Y[:, 1], '-r', label='Infected')
    plt.axis([a, b, -.1, 1.1])
    plt.legend(loc=1)
    plt.xlabel('T (days)')
    plt.ylabel('Proportion of Population')
    plt.show()
    plt.clf()


def Exercise3b():  # SIR
    beta, gamma = 1., 1. / 7.   # Exercise 3b
    a, b, ya = 0., 50., np.array([1. - (1.667e-6), 1.667e-6, 0.])

    t, Y = SIR(a, b, beta, gamma, ya)
    print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:, 1])

    plt.plot(t, Y[:, 0], '-k', label='Susceptible')
    plt.plot(t, Y[:, 2], '-b', label='Recovered')
    plt.plot(t, Y[:, 1], '-r', label='Infected')
    plt.axis([a, b, -.1, 1.1])
    plt.legend(loc=1)
    plt.xlabel('T (days)')
    plt.ylabel('Proportion of Population')
    plt.show()
    plt.clf()


def Exercise3():  # SIR
    beta, gamma = 3. / 10., 1. / 4.   # Exercise 4
    a, b, ya = 0., 500., np.array([1. - (1.667e-6), 1.667e-6, 0.])

    t, Y = SIR(a, b, beta, gamma, ya)
    print "The Maximum fraction of the population that will be infected simultaneously is", max(Y[:, 1])

    plt.plot(t, Y[:, 0], '-k', label='Susceptible')
    plt.plot(t, Y[:, 2], '-b', label='Recovered')
    plt.plot(t, Y[:, 1], '-r', label='Infected')
    plt.axis([a, b, 0., 1.])
    plt.legend(loc=1)
    plt.xlabel('T (days)')
    plt.ylabel('Proportion of Population')
    plt.show()
    plt.clf()


def Exercise4():   # measles
    a, b = 0., 1.       # Interval of the BVP
    n, N = 3, 80        # Dimension of the system/ Number of subintervals
    # Tolerance/ Maximum number of Newton steps
    TOL, Max_IT = 10. ** (-12), 40
    init_mesh = np.linspace(a, b, N + 1)  # Initial Mesh
    lmbda, mu, eta = .0279, .02, .01

    def beta1(x):
        return 1575. * (1. + np.cos(2. * np.pi * x))

    def Guess(x):
        S = .1 + .05 * np.cos(2. * np.pi * x)
        return np.array([S, 05 * (1. - S), 05 * (1. - S), .05, .05, .05])

    def ODE(x, y):
        return np.array([mu - beta1(x) * y[0] * y[2],
                         beta1(x) * y[0] * y[2] - y[1] / lmbda,
                         y[1] / lmbda - y[2] / eta,
                         0, 0, 0])

    def g(Ya, Yb):
        BCa = Ya[0:3] - Ya[3:]
        BCb = Yb[0:3] - Yb[3:]
        return BCa, BCb

    problem = bvp_solver.ProblemDefinition(num_ODE=6,
                                           num_parameters=0,
                                           num_left_boundary_conditions=3,
                                           boundary_points=(a, b),
                                           function = ODE,
                                           boundary_conditions = g)

    solution = bvp_solver.solve(problem,
                                solution_guess=Guess,
                                trace=0,
                                max_subintervals=1000,
                                tolerance=1e-9)
    Num_Sol = solution(np.linspace(a, b, N + 1))
    # Guess_array = np.zeros((6,N+1))
    # for index, x in zip(range(N+1),np.linspace(a,b,N+1)):
    # 	Guess_array[:,index] = Guess(x)
    # plt.plot(np.linspace(a,b,N+1), Guess_array[0,:] ,'-g')
    plt.plot(np.linspace(a, b, N + 1),
             Num_Sol[0, :], '-k', label='Susceptible')
    plt.plot(np.linspace(a, b, N + 1),
             Num_Sol[1, :], '-g', label='Exposed')
    plt.plot(np.linspace(a, b, N + 1),
             Num_Sol[2, :], '-r', label='Infectious')
    plt.legend(loc=5)  # middle right placement
    plt.axis([0., 1., -.01, .1])
    plt.xlabel('T (days)')
    plt.ylabel('Proportion of Population')
    plt.savefig("measles.pdf")
    plt.clf()


if __name__ == "__main__":    
    Example()
    # Exercise1()
    Exercise2()
    # Exercise2a()
    # Exercise2b()
    Exercise4()
