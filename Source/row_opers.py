from scipy import eye

def rowswap(n, j, k):
    """Swaps two rows
    
    INPUTS: n -> matrix size
            j, k -> the two rows to swap"""
    out = eye(n)
    out[j,j]=0
    out[k,k]=0
    out[j,k]=1
    out[k,j]=1
    return out
    
def cmult(n, j, const):
    """Multiplies a row by a constant
    
    INPUTS: n -> array size
            j -> row
            const -> constant"""
    out = eye(n)
    out[j,j]=const
    return out
    
def cmultadd(n, j, k, const):
    """Multiplies a row (k) by a constant and adds the result to another row (j)"""
    out = eye(n)
    out[j,k] = const
    return out
    