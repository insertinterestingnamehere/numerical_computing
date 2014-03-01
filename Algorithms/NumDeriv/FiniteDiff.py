import numpy as np

def der(fc, x, h=.0001, degree=1, type='centered', accuracy=2):
    """ Computes the numerical of the callable function 'fc at all the
    points in array 'x'. 'degree' is the degree of the derivative to be
    computed. 'type' can be 'centered', 'forward', or 'backward'.
    'accuracy' is the desired order of accuracy. For forward and backward
    differences it can take a value of 1, 2, or 3. For centered differences
    it can take a value of 2, 4, or 6."""
    # Use these lists to manage the different coefficient options.
    A = np.array([[[0., 0., -.5, 0., .5, 0., 0.],
                   [0., 1/12., -2/3., 0., 2/3., -1/12., 0.],
                   [-1/60., 3/20., -3/4., 0., 3/4., -3/20., 1/60.]],
                  [[0., 0., 1., -2., 1., 0., 0.],
                   [0., -1/12., 4/3., -5/2., 4/3., -1/12., 0.],
                   [1/90., -3/20., 3/2., -49/18., 3/2., -3/20., 1/90.]]])
    B = np.array([[[-1., 1., 0., 0., 0.],
                   [-1.5, 2., -.5, 0., 0.],
                   [-11/6., 3., -1.5, 1/3., 0.]],
                  [[1., -2., 1., 0., 0.],
                   [2., -5., 4., -1., 0.],
                   [35/12., -26/3., 19/2., -14/3., 11/12.]]])
    if type == "centered":
        acc = int(accuracy/2) - 1
    else:
        acc = int(accuracy) - 1
    if int(degree) not in [1, 2]:
        raise ValueError ("Only first and second derivatives are supported")
    if acc not in [0, 1, 2]:
        raise ValueError ("Invalid accuracy")
    if type == 'centered':
        xdifs = np.array([fc(x+i*h) for i in xrange(-3, 4)])
        return np.inner(A[degree-1,acc], xdifs.T) / h**degree
    elif type == 'forward':
        xdifs = np.array([fc(x+i*h) for i in xrange(5)])
        return np.inner(B[degree-1,acc], xdifs.T) / h**degree
    elif type == 'backward':
        xdifs = np.array([fc(x-i*h) for i in xrange(5)])
        return np.inner(B[degree-1,acc], xdifs.T) / (-h)**degree
    else:
        raise ValueError ("invalid type")
