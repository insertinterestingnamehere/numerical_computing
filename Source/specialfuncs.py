import scipy as sp
from scipy import linalg as la

def cond(matrix, P=2):
    return la.norm(matrix, P)*la.norm(la.inv(matrix), P)