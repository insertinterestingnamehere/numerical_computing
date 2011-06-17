import timeit

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
            self.gbcol="gc.enable()"
        else:
            self.gbcol="gc.disable()"

        self.funcname = func.__name__
        pfunc = lambda: func(*args, **kargs)
        self.elapsed = timeit.repeat(pfunc, self.gbcol, repeat=self.repeats, number=self.loops)
        self.runtime = min(self.elapsed)
        return [self.runtime, self.funcname]

    def printTime(self):
        result = "%s finished in %.5fs (%s loops, repeated %s times): %.5fs per loop (with %s)" % (self.funcname, self.runtime, self.loops, self.repeats, self.runtime/self.loops, self.gbcol)
        print result
