def factorial2(n):
    if n <= 0 or n%1 != 0:
        raise ValueError, "input must be a positive integer"
    
    if n == 1:
        return 1
    else:
        return n*factorial2(n-1)