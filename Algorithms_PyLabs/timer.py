import timeit

#@contextmanager
#def timer(func, *args, **kargs):
    #pfunc = partial(func, *args, **kargs)
    
    #try:
        #elapsed = timeit.repeat(pfun
        #yield
    #except:
        #raise
    #finally:
        
        
class timer(object):
    def __init__(self, repeats=3, loops=1, gc=False):
        self.repeats = repeats
        self.loops = loops
        self.gc = gc
    
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        else:
            return True
            
    def results(self):
        return self.func()
        
    def time(self, func, *args, **kargs):
        if self.gc is True:
            gbcol="gc.enable()"
        else:
            gbcol="gc.disable()"
            
        self.func = lambda: func(*args, **kargs)
        self.elapsed = timeit.repeat(self.func, gbcol, repeat=self.repeats, number=self.loops)
        mine = min(self.elapsed)
        result = "%s finished in %.5fs (%s loops, repeated %s times): %.5fs per loop (with %s)" % (func.__name__, mine, self.loops, self.repeats, mine/self.loops, gbcol)
        return result

        
def printRange(n):
    return (0 if n == 0 else
            1 if n == 1 else
            printRange(n-1)+printRange(n-2))