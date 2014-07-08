lw = np.linspace(.5, 15, 8)

for i in xrange(8):
	plt.plot(x, i*y, colours[i], linewidth=lw[i])
	
plt.ylim([-1, 8])
plt.show()