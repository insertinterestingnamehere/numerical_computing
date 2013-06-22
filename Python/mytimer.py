"""
This timer class is designed to mimic the functionality of IPython's timeit function.
"""

from collections import defaultdict
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
        self.results = defaultdict(list)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        else:
            return True

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
        print "Timing {} ...".format(funcname),
        elapsed = timeit.repeat(pfunc, self.gbcol, repeat=self.repeats, number=self.loops)
        runtime = min(elapsed)/self.loops
        print runtime
        self.results[funcname].append(runtime)