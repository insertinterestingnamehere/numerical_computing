import numpy as np

def rowswap(A, i, j):
	#swaps two rows
	#modifies A in place
	A[i], A[j] = A[j], A[i]

def cmult(A, i, const):
	#multiplies row j of A by const in place
	A[i] *= const

def cmultadd(A, i, j, const):
	#adds a constant times row k to row j
	A[i] += const*A[j]
