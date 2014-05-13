from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
n = 100
tol = .0001	
U = np.ones ((n, n))
U [:,0] = 100 # sets north boundary condition
U [:,-1] = 100 # sets south boundary condition
U [0] = 0 # sets west boundary condition
U [-1] = 0 # sets east boundary condition
# U has been changed in place (note that laplace is the name of 
# the function in this case).
laplace(U, tol) 
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface (X, Y, U, rstride=5)
plt.show()
