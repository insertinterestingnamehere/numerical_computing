def fib2(n):
    try:
        for k in range(len(fib2.f)+1, n):
            fib2.f.append(f[k-2]+f[k-1])
        return fib2.f
    except AttributeError:
        fib2.f = [1,1]





    try:
        f = fib2.f
    except:
        fib2.f=f=[1,1]
        
    for k in range(len(f)+1, n):
        #f[k] = f[k-2]+f[k-1]
        f.append(f[k-2]+f[k-1])
    
    return f