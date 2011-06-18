from scipy import sp

def op1(A,j,k):
    temp = A[j,:]
    A[j,:] = A[k,:]
    A[k,:] = temp
    return temp
    
