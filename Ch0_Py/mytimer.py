"""
This timer class is designed to mimic the functionality of IPython's timeit function.
"""

import timeit

class timer(object):
    def __init__(self, repeats=3, loops=5, gc=False):
        """
        Initialize the timer class.
                       
        repeats - How many times to call timeit.  Each repeat 
        reinitializes the execution environment
                              
        loops - How many loops to run when timing.
                                  
        gc - Toggles Python's garbage collector.  Default=False (do not collect garbage)
        """
        self.repeats = repeats
        self.loops = loops
        self.gc = gc
        self.results = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        else:
            return True

    def results(self):
        """
        Return a list of timing results
        """
        return self.results

    def time(self, func, *args, **kargs):
        """
        Time a function using timeit.  The result of each execution of this 
        function is stored in a list along with the parameters of the function.
        This list can be retreived with the result() method.                                        
        func - a function handle to the code being timed
        args - the input arguments to the function
        kargs - the input keyword arguments to the function
        """
        if self.gc is True:
            self.gbcol="gc.enable()"
        else:
            self.gbcol="gc.disable()"

        funcname = func.__name__
        if funcname == "<lambda>":
            funcname = func.__repr__()
        pfunc = lambda: func(*args, **kargs)
        print "Timing %s ..." % funcname
        elapsed = timeit.repeat(pfunc, self.gbcol, repeat=self.repeats, number=self.loops)
        runtime = min(elapsed)
        
        if funcname in self.results:
            self.results[funcname].append(runtime)
        else:
            self.results[funcname] = [runtime]

    def printTimes(self):
        """
        Print the results to stdout in an easy to read fashion
        """
        print '\n'.join(["%s finished in %.5fs (%s loops, repeated %s times): %.5fs per loop (with %s)" % (f[1], f[0], self.loops, self.repeats, f[0]/self.loops, self.gbcol) for f in self.results])
