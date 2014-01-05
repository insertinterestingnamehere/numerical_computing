import numpy as np
from itertools import islice, izip

class InfeasibleSystem(Exception):
    pass

class UnboundedSystem(Exception):
    pass

class MaximalValue(Exception):
        pass

class SimplexSolver(object):
    r'''Solve a standard maximization problem using the Simplex algorithm.'''
    
    isinf = np.math.isinf
    isnan = np.math.isnan
    
    def solve(self):
        #ignore division by zero errors in NumPy
        #These can occur in the Simplex algorithm
        nperrs = np.seterr(divide='ignore')
        try:
            while self.systemState is None:
                self._pivot()
        except MaximalValue as ex:
            return ex.message
        finally:
            #set NumPy error settings back to what they were
            np.seterr(**nperrs)
        
        
    def generateTableau(self):
        r'''Generate the simplex tableau for solving the system
        
        [0,0] is the value of the objective function
        Along the top row of the tableau is the objective function
        in the form 0=-c
        
        Down the first column are the system's current constraints.
        The non-negative constraints for all variables is assumed
        
        The remaining entries in the matrix are the constraint equations
        '''
        self._generateTableau(self.c, self.A, self.b)
    
    def __init__(self, c, A, b):
        #store the original system to make tweaking the system easier
        self.c = np.asarray(c)
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        
        self.systemState = None
        
        #check the program dimensions
        if (self.b.size, self.c.size) != self.A.shape:
            raise ValueError("Incompatible dimensions!")
        
        #check the system for feasibility
        if (self.b < 0).any():
            self._auxiliary_program()
        else:
            self.generateTableau()
        
    def _generateTableau(self, c, A, b):
        nr, nc = A.shape
        self.tab = np.zeros((nr+1,nc+nr+1))
        
        _t = nc + 1
        self.tab[0,1:(_t)] = -c
        self.tab[1:,0] = b
        self.tab[1:,1:(_t)] = A
        self.tab[1:,(_t):] = np.eye(nr)
        
        self.vars = range(nc, nc+nr)+range(0, nc)
        self.nbasic = nr
        
    def _auxiliary_program(self):
        r'''Setup the auxiliary linear program.
        This method is used in the case that the origin is not feasible.
        This method will perform the first pivot and find a feasible point if any exists and leave the program in a state that the solve method can use.  Otherwise an raise an InfeasibleSystem exception.'''
        
        nr, nc = self.A.shape
        
        #setup auxilliary tableau
        c = np.array([-1]+[0]*nc)
        A = np.empty((nr, nc+1))
        A[:,0] = -1
        A[:,1:] = self.A
        
        print A
        self._generateTableau(c, A, self.b)
        print self.tab
        #manually perform first pivot on x_0
        #find the most negative constraint
        row_pivot = self.b.argmin()
        x0 = self.vars.index(0)
        print "Vars: ",self.vars
        print "Pivot row: ",self.vars[row_pivot]
        print "Pivot col: ",self.vars[x0]
        self.vars[x0], self.vars[row_pivot] = self.vars[row_pivot], self.vars[x0]
        print "new vars: ",self.vars
        self._reduceform(row_pivot, 0)
        print "First pivot"
        print self.tab
        #continue with regular pivoting
        print self.getState()
        print "Solving..."
        
        #remove the x0 variable
        #self.vars.remove(max(self.vars))
        #self.tab = np.delete(self.tab, 1, axis=1)
        
        #csize = self.c.size
        #self.tab[0, 1:csize+1] = -self.c
        #print self.tab
        #self.solve()
        #print self.tab
        #if self.systemState is MaximalValue:
            #return self.getState()
        #else:
            #self.systemState = InfeasibleSystem()
            #raise self.systemState
        
    def _pivot_col(self):
        '''Determine the index of the next pivot column
        
        The pivot column is the first positive coeff of the objective function (Bland's Rule)'''
        
        for i, j in enumerate(self.tab[0][1:], 1):
            if j < 0:
                return i
            
    def _zip_basic(self, seq):
        return izip(islice(self.vars, self.nbasic), seq)
        
    def _pivot_row(self, col):
        '''Determine the index of the next pivot row'''
        
        #do a ratio test to find smallest index non-negative ratio
        ratios = self.tab[1:,0]/self.tab[1:,col]
        
        #python version of argsort
        basic_vars = self.vars[:self.nbasic]
        min_x = sorted(range(len(basic_vars)), key=basic_vars.__getitem__)
        for x in min_x:
            t = ratios[x]
            if not (self.isinf(t) or self.isnan(t)) and t >= 0:
                return x + 1   
        
    def _pivot(self, row=None, col=None):
        r'''Perform a pivot in the tableau'''
        
        #find what column to pivot
        col = col or self._pivot_col()
        if col is None:
            #no improvement can be made
            self.systemState = MaximalValue(self.getState())
            raise self.systemState
        
        if self.tab[0, col] < 0:
            #we have not yet reached a maximal value
            #check if unbounded
            if (self.tab[1:,col] > 0).any():
                if row is None:
                    row = self._pivot_row(col)

                #swap the entering and exiting variables
                #swap basic and nonbasic variables
                icol = self.vars.index(col-1)
                self.vars[icol], self.vars[row-1] = self.vars[row-1], self.vars[icol]
                                
                #reduce the tableau using the element at (row, col) as the pivot
                self._reduceform(row, col)
            else:
                self.systemState = UnboundedSystem('Unbounded System!')
                raise self.systemState
        else:
            self.systemState = MaximalValue(self.getState())
            raise self.systemState
            
    def _reduceform(self, row, col):
        r'''Reduces col to elementary vector using where row has the unitary element'''
        
        #divide row by tab[row,col]
        #this gets the one
        self.tab[row] /= self.tab[row, col]
        
        #reduce the rest of the column to zeros
        _row = self.tab[row]
        for i in xrange(self.tab.shape[0]):
            if i != row:
                self.tab[i] -= _row*self.tab[i, col]
            
        
    def __repr__(self):
        return "Simplex Solver: Maximize {}\nSubject to\n{}\nConstraints: {}".format(self.c, self.A, self.b)
    
    def __str__(self):
        return str(self.tab)
        
    def getState(self):
        r'''Return the current state of the linear program.
        
        A tuple is returned containing:
            0:  the current value of the objective function
            1:  a dictionary of the basic variables (numbered according to variable subscripts)
            2:  a dictionary of the nonbasic variables (also numbered according to variable subscripts)
        '''
        
        b = dict(self._zip_basic(self.tab[1:,0]))
        nb = {k:0 for k in self.vars[self.nbasic:]}
        
        return self.tab[0,0], b, nb