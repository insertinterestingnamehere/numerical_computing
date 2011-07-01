#!/usr/bin/python
# -*- coding: utf-8 -*-


def sum_loop(n):
    """Sum all the numbers between 0 and n using a for loop"""

    sum = 0
    for i in range(n):
        if i % 2 == 1:
            sum += i
    return sum

def sum_range(n):
    return sum(range(1,n,2))

def sum_list(n):
    return sum([i for i in range(n) if i % 2 == 1])


def expSS(x, i=100):
    """Calculate e**x by scaling and squaring"""

    from scipy import factorial

    results = []
    for val in x:
        (t, r) = (0, 0)
        val = float(val)
        while val >= 10:
            val = val / 2
            t += 2.0

        # calculate sum
        results.append(sum([val ** z / factorial(z) for z in range(i)])
                       ** t)
    return results


