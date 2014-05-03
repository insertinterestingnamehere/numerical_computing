import timeit
import pickle
import collections

def time_func(f, args=(), kargs={}, repeat=3, number=100):
    pfunc = lambda: f(*args, **kargs)
    T = timeit.Timer(pfunc)

    try:
        _t = T.repeat(repeat=repeat, number=int(number))
        runtime = min(_t)/float(number)
        return f.__name__, runtime
    except:
        T.print_exc()

def rot_list(l):
    for i in xrange(len(l)):
        l.insert(0, l.pop())

def rot_deque(d):
    for i in xrange(len(d)):
        d.appendleft(d.pop())

