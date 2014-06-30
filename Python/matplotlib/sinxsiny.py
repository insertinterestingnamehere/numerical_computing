import numpy as np 
from matplotlib import pyplot as plt 
n = 401 
x = np.linspace(-6, 6, n) 
y = np.linspace(-6, 6, n) 
X, Y = np.meshgrid(x, y) # returns a coordinate matrix given coordinate vectors. 
C = np.sin(X) * np.sin(Y) 
plt.pcolormesh(X, Y, C)
plt.show()