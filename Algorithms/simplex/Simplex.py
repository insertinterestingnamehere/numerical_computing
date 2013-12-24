import numpy as np

class InfeasibleSystem(Exception):
    pass

class UnboundedSystem(Exception):
    pass

class SimplexSolver(object):
    
    class _MaximalValue(Exception):
        pass
    
    def __init__(self, c, A, b):
        self.c = np.asarray(c)
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        
        #check the system for feasibility
        if (b.size, c.size) != A.shape:
            raise InfeasibleSystem
            
        self.generateTab()
        
    def generateTab(self):
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
        #self.basic = range(nc, nc+nr)
        #self.nonbasic = range(0, nc)
        
    def _pivot_col(self):
        '''Determine the index of the next pivot column
        
        Do so by finding the most negative coefficient in the objective function.'''
        
        return self.tab[0,1:].argmin()
        
    def _pivot_row(self, col):
        '''Determine the index of the next pivot row'''
        
        #do a ratio test to find smallest non-negative ratio
        
        ratios = self.tab[1:,0]/self.tab[1:,col+1]
        print ratios
        ratios[ratios<0] = np.inf
        return ratios.argmin()
    
    def _pivot(self):
        col = self._pivot_col()        
        if self.tab[0, 1+col] < 0:
            #check if unbounded
            if (self.tab[1:,1+col] > 0).any():
                row = self._pivot_row(col)

                #swap basic and nonbasic variables
                icol, irow = self.vars.index(col), self.vars.index(row)
                print "Vars index {} {}".format(icol, irow)
                self.vars[icol], self.vars[row] = self.vars[row], self.vars[icol]
                
                print self.vars
                self._reduceform(row, col)
            else:
                raise UnboundedSystem()
        else:
            raise self._MaximalValue(self.getState());
            
    def _reduceform(self, row, col):
        '''Reduce the col to elementary vector using where row has the 1 element'''
        
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
        b = dict(zip(self.vars[:self.nbasic], self.tab[1:,0]))
        nb = {k:0 for k in self.vars[self.nbasic:]}
        
        return self.tab[0,0], b, nb