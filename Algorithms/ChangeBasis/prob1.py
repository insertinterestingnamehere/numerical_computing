from matplotlib import pyplot as plt
# Assume the arrays 'old' and 'new' have x values along the first row and y values along the second row.

plt.subplot(2, 1, 1)
plt.scatter(old[0], old[1])
plt.axis('equal')
plt.subplot(2, 1, 2)
plt.scatter(new[0], new[1])
plt.show()