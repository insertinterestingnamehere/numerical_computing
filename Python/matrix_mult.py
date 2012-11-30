k = 10
a = [range(i, i+k) for i in range(0, k**2, k)]

def square(l):
    rows = len(l)
    cols = len(l[0])
    squared = [[0]*cols for i in xrange(rows)]
    
    for i in xrange(rows):
        for j in xrange(cols):
            res = 0
            for k in xrange(cols):
                res += l[i][k] * l[k][j]
            squared[i][j] = res

    return squared
