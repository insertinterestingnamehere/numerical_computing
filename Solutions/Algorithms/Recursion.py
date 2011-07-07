
import scipy as sp

def coeffExpand(M):
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    M = sp.atleast_2d(M)
    if M.shape == (1,1):
        return M.item(0,0)
    else:
        det = sum(M[0,i]*(((-1)**i)*coeffExpand(sp.delete(M[1:],i,1))) for i in range(len(M)))
        return det
