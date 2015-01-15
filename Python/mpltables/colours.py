import numpy as np
from matplotlib import pyplot as plt

colours = np.array(["k", "g", "b", "r", "c", "m", "y", "w"])
x = np.linspace(0, 5, 1000)
y = np.ones(1000)

for i in xrange(8):
	plt.plot(x, i*y, colours[i])
	
plt.ylim([-1, 8])
plt.show()