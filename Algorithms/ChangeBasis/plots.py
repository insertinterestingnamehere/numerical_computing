import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from matplotlib import pyplot as plt
import solutions
import numpy as np

pts = .2*np.load('horse.npy')[:,::25] 


def plotOldNew(old, new, label):
    ''' 
    This plotting script gives better results than the one provided in the lab
    text. Please use this to plot your figures.
    Inputs:
    new -- a (2,n) numpy array containing the transformed x-coordinates on the 
            first row, y-coordinates on the second row.
    old -- a (2,n) numpy array containing the original x-coordinates on the first
            row, y-coordinates on the second row.
    '''
    # Find the largest and smallest x- and y-values, used to set the size of the axes
    new_max, old_max = new.max(axis=1), old.max(axis=1)
    new_min, old_min = new.min(axis=1), old.min(axis=1)
    
    x_max = max((new_max[0], old_max[0])) + 1
    x_min = min((new_min[0], old_min[0])) - 1
    y_max = max((new_max[1], old_max[1])) + 1
    y_min = min((new_min[1], old_min[1])) - 1

    print x_min, x_max
    print y_min, y_max
    # Create the first subplot
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(old[0], old[1], 'k,')
    ax1.axis('equal')
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylabel("Original")

    # Create the second subplot
    ax2.plot(new[0], new[1], 'k,')
    ax2.axis('equal')
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylabel(label)
    return f
    
def stretch():
    fig = plotOldNew(pts, solutions.dilate(pts, np.array([1.5, 1.5])), "Stretch")
    fig.savefig("stretch.pdf")
    plt.close(fig.number)

def rotate():
    fig = plotOldNew(pts, solutions.rotate(pts, np.pi/3.), "Rotate")
    fig.savefig("rotate.pdf")
    plt.close(fig.number)
    
def shear():
    fig = plotOldNew(pts, solutions.shear(pts, 1.02, 0), "Shear")
    fig.savefig("shear.pdf")
    plt.close(fig.number)
    
def reflect():
    fig = plotOldNew(pts, solutions.reflect(pts, np.sqrt(np.array([3, 1]))), "Reflect")
    fig.savefig("reflect.pdf")
    plt.close(fig.number)
    
def translate():
    fig = plotOldNew(pts, solutions.translate(pts, np.array([2, 0])), "Translate")
    fig.savefig("translate.pdf")
    plt.close(fig.number)
    
def combo():
    p = solutions.shear(pts, -1.02, .5)
    p = solutions.translate(p, np.array([-2, .5]))
    p = solutions.reflect(p, np.array([-2, .5]))
    
    fig = plotOldNew(pts, p, "General Affine")
    fig.savefig("combo.pdf")
    plt.close(fig.number)


if __name__ == "__main__":
    stretch()
    rotate()
    shear()
    reflect()
    translate()
    combo()