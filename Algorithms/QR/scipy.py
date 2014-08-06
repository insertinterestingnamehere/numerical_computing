import numpy as np
from scipy import linalg as la
A = np.random.rand(4,3)
Q, R = la.qr(A)
Q.dot(R) == A                      # there are False entries
np.allclose(Q.dot(R), A)           # A = QR
np.allclose(Q.T.dot(Q), np.eye(4)) # Q is indeed, orthogonal
