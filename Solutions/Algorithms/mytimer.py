import timeit

class timer(object):
    def __init__(self, repeats=3, loops=1, gc=False):
        self.repeats = repeats
        self.loops = loops
        self.gc = gc
        self.results = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        else:
            return True

    def results(self):
        return self.results

    def time(self, func, *args, **kargs):
        if self.gc is True:
            self.gbcol="gc.enable()"
        else:
            self.gbcol="gc.disable()"

        pfunc = lambda: func(*args, **kargs)
        print "Timing %s ..." % func.__name__
        elapsed = timeit.repeat(pfunc, self.gbcol, repeat=self.repeats, number=self.loops)
        runtime = min(elapsed)
        self.results.append((runtime, func.__name__))

    def printTimes(self):
        print '\n'.join(["%s finished in %.5fs (%s loops, repeated %s times): %.5fs per loop (with %s)" % (f[1], f[0], self.loops, self.repeats, f[0]/self.loops, self.gbcol) for f in self.results])
