import numpy as np

# problem 1
class Precise(object):
    def __init__(self, val=0, exp=0):
        self.val = val
        # note, I'm storing the power of the last significant digit.
        self.sig = exp - len(str(self.val)) +1
        
        
    def exp(self):
        # get the exponent from the sig value.
        return len(str(self.val)) - 1 + self.sig
    
    
    def pyfloat(self):
        # return a python float from the floating point object.
        return self.val * 10**(self.sig)
    
    
    def copy(self):
        # make a copy of the floating point object.
        new = precise(val=self.val)
        new.sig = self.sig
        return new
    
    
    def __repr__(self):
        # print method
        s = str(self.val)
        return s[0] + '.' + s[1:] + " x 10^" + str(self.exp())
    
    
    def __add__(self, other):
        # addition
        if self.sig < other.sig:
            new = self.copy()
            new.val *= 10**(other.sig - self.sig)
            new.sig = self.sig
            new.val += other.val
        elif other.sig < self.sig:
            new = other.copy()
            new.val *= 10**(self.sig - other.sig)
            new.sig = other.sig
            new.val += self.val
        else:
            new = other.copy()
            new.val += self.val
        return new
    
    
    def __mul__(self, other):
        # multiplication
        new = self.copy()
        new.sig += other.sig
        new.val *= other.val
        return new
    
    
    def __sub__(self, other):
        # subtraction
        new = other.copy()
        new.val *= -1
        return self + new
    
    
    def truncate(self, newsig):
        # truncation
        if self.sig < newsig:
            self.val = int(str(self.val)[:(self.sig-newsig)])
            self.sig = newsig
 
class Significant(object):
    # most of these methods are based on the previous class
    def __init__(self, val=Precise(), err=Precise()):
        self.val = val
        # note: error should ALWAYS be positive
        self.err = err
    
    
    def pyfloat(self):
        return self.val.pyfloat()
    
    
    def copy(self):
        new = significant()
        new.val = self.val.copy()
        new.err = self.err.copy()
        return new
    
    
    def __repr__(self):
        return str(self.val) + " +- " + str(self.err)
    
    
    def __add__(self, other):
        new = self.copy()
        new.val += other.val
        new.err += other.err
        return new
    
    
    def __sub__(self, other):
        new = self.copy()
        new.val -= other.val
        new.err += other.err
        return new
    
    
    def __mul__(self, other):
        new = self.copy()
        new.val *= other.val
        new.err  = self.err*other.val + self.val*other.err + self.err*other.err
        return new
    
    
    def truncate(self):
        if self.err.val == 1:
            self.err.sig = self.err.exp()
            self.val.truncate(self.err.sig)
        elif self.err.val != 0:
            self.err.sig = self.err.exp() + 1
            self.err.val = 1
            self.val.truncate(self.err.sig)

# problem 2
def sqrt64(A, reps):
    Ac = A.copy()
    I = Ac.view(dtype=np.int64)
    I >>= 1
    I += (1<<61) - (1<<51)
    for i in xrange(reps):
        Ac = .5 *(Ac + A / Ac)
    return Ac

# For the analysis, it is system dependent, but the sqrt here will probably be faster for low accuracy.
# Again, for the fast inverse square root, it is system dependent, but the cython version here
# should still be moderately faster than the version using the sqrt function in C for up to 4 iterations
# which is just as accurate.
