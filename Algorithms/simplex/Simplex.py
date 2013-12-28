import numpy as np

class InfeasibleSystem(Exception):
    pass

class UnboundedSystem(Exception):
    pass

class MaximalValue(Exception):
        pass

class SimplexSolver(object):
    r'''Solve a standard maximization problem using the Simplex algorithm.'''
    
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
        nr, nc = self.A.shape
        self.tab = np.zeros((nr+1,nc+nr+2))
        
        _t = 1 + nc
        self.tab[0,1:(_t)] = -self.c
        self.tab[0,-1] = 1
        self.tab[1:,0] = self.b
        self.tab[1:,1:(_t)] = self.A
        self.tab[1:,(_t):-1] = np.eye(nr)
        
        self.vars = range(nc, nc+nr)+range(0, nc)
        self.nbasic = nr
    
    def __init__(self, c, A, b):
        #store the original system to make tweaking the system easier
        self.c = np.asarray(c)
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        
        self.systemState = None
        
        #check the system for feasibility
        if (b.size, c.size) != A.shape:
            self.systemState = InfeasibleSystem('The system is infeasible')
            raise self.systemState
            
        self.generateTableau()
        
    def _pivot_col(self):
        '''Determine the index of the next pivot column
        
        Do so by finding the most negative coefficient in the objective function.'''
        
        return self.tab[0,1:].argmin()
        
    def _pivot_row(self, col):
        '''Determine the index of the next pivot row'''
        
        #do a ratio test to find smallest non-negative ratio
        ratios = self.tab[1:,0]/self.tab[1:,col+1]
        ratios[ratios<0] = np.inf
        return ratios.argmin()
    
    def _pivot(self):
        r'''Perform a pivot in the tableau'''
        
        #find what column to pivot
        col = self._pivot_col()        
        if self.tab[0, 1+col] < 0:
            #we have not yet reached a maximal value
            #check if unbounded
            if (self.tab[1:,1+col] > 0).any():
                row = self._pivot_row(col)

                #swap the entering and exiting variables
                #swap basic and nonbasic variables
                icol = self.vars.index(col)
                self.vars[icol], self.vars[row] = self.vars[row], self.vars[icol]
                
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
        
        row1 = row + 1
        col1 = col + 1
        #divide row by tab[1+row,1+col]
        #this gets the one
        self.tab[row1] /= self.tab[row1, col1]
        
        #reduce the rest of the column to zeros
        _row = self.tab[row1]
        for i in xrange(self.tab.shape[0]):
            if i == row1:
                continue
            self.tab[i] -= _row*self.tab[i, col1]
            
        
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
        
        b = dict(zip(self.vars[:self.nbasic], self.tab[1:,0]))
        nb = {k:0 for k in self.vars[self.nbasic:]}
        
        return self.tab[0,0], b, nb