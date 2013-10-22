import numpy as np
from matplotlib import pyplot as plt

def shiftByMean(A):
    '''
    Shift the columns of the input array by their respective means.
    Inputs:
        A -- an (m,n) array
    Return:
        a (m,n) array whose columns are the mean-shifted counterparts to A.
    '''
    return A - A.mean(axis=0)

def computeVariance(A):
    '''
    Calculate the variance of each column of the input array.
    Inputs:
        A -- an (m,n) array, not necessarily mean-shifted.
    Return:
        a 1-d array with n entries, each giving the variance of the corresponding column.
    '''
    return (shiftByMean(A)**2).sum(axis=0)/np.float(A.shape[0])

def reportStDev():
    '''
    Print the answer to the question in problem 1.
    You may also include the necessary code to import and process the data, but this
    need not necessarily be in the body of this function.
    '''
    data = np.genfromtxt('weight_age_fat.txt', skip_header=1, usecols=[2,3,4])
    col_titles = ['weight', 'age', 'blood fat content']
    stds = np.sqrt(computeVariance(data))
    ind = np.argmin(stds)
    min_std = np.min(stds)
    print col_titles[ind], min_std

def corrMatrix(W):
    '''
    Compute the correlation matrix for the columns of input array W
    '''
    X = W - W.mean(axis=0)
    Y = X/np.sqrt((X**2).sum(axis=0))
    return np.dot(Y.T, Y)
    
def reportCorr():
    '''
    Print the answers to the questions in problem 3. Ther are three distinct questions,
    so there should be three distinct print statements, indicating which question is
    being answered. Report the columns as their number in the original data file. The first
    column, which we omitted, would be column 0, and the last column (giving the mortality)
    would be column 16.
    Finally, plot the data as described in the problem statement.
    '''
    data = np.genfromtxt('mortality.txt', skip_header=17, usecols=range(1,17))
    corr_mat = corrMatrix(data)

    # find the column whose correlation with the last column is closest to zero
    min_col = np.argmin(np.abs(corr_mat[:,-1]))
    print "The column most nearly uncorrelated with mortality rate is",  min_col + 1

    # now find the pair of distinct columns with highest correlation.
    # set diagonals to zero, since we want to consider only distinct columns
    np.fill_diagonal(corr_mat,0)
    max_rows = corr_mat.max(axis=1)
    max_row = np.argmax(max_rows)
    max_col = np.argmax(corr_mat[max_row,:])
    print "The pair of columns with largest correlation is", (max_row+1,max_col+1)

    # find column with highest correlation to mortality
    min_mort = np.argmin(corr_mat[:,-1])
    print "The column that is most negatively correlated with mortality is", min_mort + 1

    plt.subplot(311)
    plt.scatter(data[:,max_row], data[:, max_col])
    plt.subplot(312)
    plt.scatter(data[:,min_mort], data[:,-1])
    plt.subplot(313)
    plt.scatter(data[:,min_col], data[:,-1])
    plt.show()
    plt.clf()
