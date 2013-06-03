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

def pickle_obj():
    j = ['5', 4, 2, 1, 'Python', 99918]

    with open('out.pkl', 'w') as f:
        pickle.dump(f, j)

    with open('out.pkl', 'r') as f:
        a = pickle.load(f)

    print a == j

def rot_list(l):
    for i in xrange(len(l)):
        l.insert(0, l.pop())

def rot_deque(d):
    for i in xrange(len(d)):
        d.appendleft(d.pop())

