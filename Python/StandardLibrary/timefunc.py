import timeit

def time_func(f, args=(), kargs={}, repeat=3, number=100):
    pfunc = lambda: f(*args, **kargs)
    T = timeit.Timer(pfunc)

    try:
        _t = T.repeat(repeat=repeat, number=int(number))
        runtime = min(_t)/float(number)
        return {f.__name__: runtime}
    except:
        T.print_exc()