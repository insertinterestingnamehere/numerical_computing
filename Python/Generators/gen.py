
def gray_combinations(n):
    t = [0]*n
    yield t

    final = [1]+[0]*(n-1)
    while t != final:
        if sum(t)%2 == 0:
            t[-1] = (t[-1]+1) % 2
        else:
            for i, j in enumerate(reversed(t), 1):
                if j:
                    t[-i-1] = (t[-i-1]+1)%2
                    break

        yield t
