from itertools import izip, compress

def bin_combinations(n):
    '''Generate the sets of a powerset by counting in binary.'''
    s = len(n)

    #start at 0 (the empty set) and count to 2**s
    #Each element is assigned a bit position.
    #If the bit in that position is 1, then include in set, otherwise ignore
    b = 0
    while b < 2**s:
        #check for ones in the binary representation of b
        yield [i for i, v in izip(n, reversed('{0:b}'.format(b))) if v == '1']
        b += 1

def binary_gray(seq):
    """
    Calculates the gray code using binary.
    Returns an element and a boolean.
    If True, element should be added to make next set.
    If False, element should be removed to make next set.
    """
    pool = tuple(seq)
    
    #We are subtracting one from the bit_length, one from the index
    n = len(pool) - 1
    
    #initialize to 0.  This is represents the empty set
    #We are manipulating bits directly with >> and << bitshifts
    old = (0 >> 1) ^ 0
    for i in xrange(1, 2**(n + 1)):
        new = (i >> 1) ^ i        
        #do old XOR new to find just the changed bit
        #bit_length() will return the position of the highest set bit
        ind = (old ^ new).bit_length() - 1
        # find the action with a mask.  Nonzero if we should add, zero if we remove
        yield pool[ind], bool(new & (1 << ind))
        old = new
        
def gray_combinations(n):
    """Return each gray code.  
    Student solutions are expected to be similiar to this method
    """
    s = len(n)
    t = [0]*s
    
    #return the empty set
    yield list(compress(reversed(n), t))

    #precompute final gray code. 10...0
    final = [1]+[0]*(s-1)
    while t != final:
        if sum(t) % 2 == 0:
            #if even number of ones, toggle last element
            t[-1] = (t[-1]+1) % 2
        else:
            #otherwise find right-most one digit and toggle the left element
            for i, j in enumerate(reversed(t), 1):
                if j:
                    t[-i-1] = (t[-i-1]+1)%2
                    break
        #yield the current gray code
        yield list(compress(reversed(n), t))
