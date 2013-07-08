class ComplexNumber(object):
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real+other.real, self.imag+other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real-other.real, self.imag-other.imag)

    def __mul__(self, other):
        r = self.real*other.real-(self.imag*other.imag)
        i = self.real*other.imag+self.imag*other.real
        return ComplexNumber(r, i)
        
    def __div__(self, other):
        conj = other.conjugate()

        numer = self*conj
        denom = float((other*conj).real)
        return ComplexNumber(numer.real/denom, numer.imag/denom)

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def __repr__(self):
        return "{}{:+}i".format(self.real, self.imag)

    def norm(self):
        return ComplexNumber((self.real**2+self.imag**2)**.5, 0)

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag
    
    def dist(self, other):
        return self.norm() - other.norm()