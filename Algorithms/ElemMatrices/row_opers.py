import numpy as np

def rowswap(A, j, k):
	#swaps two rows
	#modifies A in place
	A[j], A[k] = A[k], A[j]

def cmult(A, j, const):
	#multiplies row j of A by const in place
	A[j] *= const

def cmultadd(A, j, k, const):
	#adds a constant times row k to row j
	A[j] += const*A[k]
