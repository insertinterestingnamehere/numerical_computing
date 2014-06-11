import numpy as np 
from matplotlib import pyplot as plt 
x = np.linspace(-np.pi, np.pi, 400) 
y1 = np.sin(x) 
y2 = np.cos(x)
plt.subplot(211) 
plt.plot(x, y1) 
plt.subplot(212) 
plt.plot(x, y2)
plt.show() 