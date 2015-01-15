'''
Solutions file for Volume 1, Lab 14
Least Squares fitting
'''

# include all necessary imports

def fitLine():
    '''
    See problem description.
    '''
    # load in the data
    with np.load("data.npz") as data:
        arr = data['linepts']
    
    # create A
    A = np.vstack((arr[:,0], np.ones(arr.shape[0]))).T
    
    # obtain least square solution
    xhat = np.dot(np.dot(la.inv(np.dot(A.T,A)),A.T),arr[:,1])
    
    # plot results
    x0 = np.linspace(arr[:,0].min(), arr[:,0].max(), 100)
    y0 = xhat[0]*x0 + xhat[1]
    plt.plot(arr[:,0], arr[:,1], '*', x0, y0)
    plt.show()


def fitCircle():
    '''
    See problem description.
    '''
    with np.load('data.npz') as data:
        circle = data['circlepts']
    A = np.hstack((2*circle, np.ones((circle.shape[0],1))))
    b = (circle**2).sum(axis=1)
    x = np.dot(np.dot(la.inv(np.dot(A.T,A)),A.T),b)
    c1,c2,c3 = x
    r = sqrt(c1**2 + c2**2 + c3)
    theta = np.linspace(0,2*np.pi,200)
    plt.plot(r*np.cos(theta)+c1,r*np.sin(theta)+c2,'-',circle[:,0],circle[:,1],'*')
    plt.show()

def fitEllipse(P):
    '''
    See problem description.
    Inputs:
        P -- (m,2) shape array representing x- and y-coordinates.
    Returns:
        x -- the least squares solution of the problem.
    '''
    b = np.ones((P.shape[0],1), dtype=np.float)
    A = np.hstack((P,P**2, P.prod(axis=1, keepdims=True)))
    x = np.dot(np.dot(la.inv(np.dot(A.T,A)),A.T),b)
    b, d, a, e, c = x
    return np.array([a,b,c,d,e])
