from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
n = 100
tol = .0001	
U = np.ones ((n, n))
U [:,0] = 100 # set north boundary condition
U [:,-1] = 100 # set south boundary condition
U [0] = 0 # set west boundary condition
U [-1] = 0 # set east boundary condition
laplace(U, tol) # U has been changed in place (note that laplace is the name of the function, here)
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface (X, Y, U, rstride=5)
plt.show()
