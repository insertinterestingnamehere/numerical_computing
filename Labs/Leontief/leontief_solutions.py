'''
Lab 10: Leontieff Input-Output Models
Solutions File
'''

import numpy as np
from scipy import linalg as la

def geomSeries(C, d, m):
    '''
    Calculate the geometric series of C up to the m power, multiplied by d.
    Inputs:
        C -- a (n,n) shape array
        d -- a length n array
        m -- a nonnegative integer
    Return:
        The value (\sum_{i=0}^m C^i)d
    '''
    powC = C.copy()
    sumC = np.eye(C.shape[0]) + powC
    for k in xrange(m):
        powC = C.dot(powC)
        sumC += powC
    return np.dot(sumC, d)

def construction():
    '''
    Calculate the cost vector for producing an additional 50% of construction.
    Calculate an approximation by using geomSeries for m=5.
    Then calculate the exact answer.
    Return both answers, with the approximation first.
    '''
    IO = np. array ([[250. , 150. , 30. , 600.] , [25. , 25. , 20. , 280.] ,[50. , 20. , 5., 120.]])
    IOCoeff = IO [: ,:3] / IO [: ,3]
    d = np.array([0., 0., 60.0]) # a 50% increase over a baseline 120 means 60 additional units

    return geomSeries(IOCoeff, d, 5), np.dot(la.inv(np.eye(IOCoeff.shape[0])-IOCoeff), d)

def demand():
    '''
    Calculate and return the demand vector for the three product economy.
    '''
    IO = np. array ([[250. , 150. , 30. , 600.] , [25. , 25. , 20., 280.] ,[50. , 20. , 5., 120.]])
    IOCoeff = IO [: ,:3] / IO [: ,3]
    X = np.array([600,280,120]) # this is the total output, as given in the lab

    return X - np.dot(IOCoeff,X)

def cityOutput():
    '''
    Calculate and return the required output vector for the city, given the
    demand vector.
    '''
    C = np.array([[.2,.3,.3],[.1,.2,.3],[.2,.2,.2]]) # IO coefficients
    D = np.array([100000,100000,40000]) # demand vector
    return np.dot(la.inv(np.eye(3)-C),D) # from the formula in the lab

def getIOCoeffs():
    '''
    Import the data from the csv file as described in the problem statement.
    Calculate and return the IO coefficient matrix as well as the demand vector.
    Return both of these values, the matrix first, and then the vector.
    '''
    data = np.genfromtxt("io2002table.txt", delimiter='\t', skiprows=1,
                         usecols=np.arange(1,52), missing_values = [''],
                         filling_values={i:0 for i in xrange(1,52)})
    iocoeff = data[:,:-1] / data[:,-1]
    X = data[:,-1] # this is the last column, which corresponds to total output
    D = X - np.dot(iocoeff, X) # this is our demand vector
    return iocoeff, D


def washingtonOutput(iocoeffs, demand):
    '''
    Calculate the output vector corresponding to an increase of 10% in the demand
    for construction.
    Inputs:
        iocoeffs -- the input-output coefficient matrix for the Washington economy, as
                    calculated in the getIOCoeffs function.
        demand -- the demand vector for the Washington economy, as calculated in the
                  getIOCoeffs function.
    Return:
        The output vector corresponding to an increase of 10% in the construction demand.
    '''
    # now increase the construction demand by 10% (entry 8)
    demand[8] += demand[8]*.10
    # now calculate the new output vector corresponding to this demand:
    return la.solve(np.eye(iocoeffs.shape[0]) - iocoeffs, demand)
