def type_I(A, i, j):
	# Swap two rows
	A[i], A[j] = np.copy(A[j]), np.copy(A[i])

def type_II(A, i, const):
	# Multiply row j of A by const
	A[i] *= const

def type_III(A, i, j, const):
	# Add a constant times row j to row i
	A[i] += const*A[j]
