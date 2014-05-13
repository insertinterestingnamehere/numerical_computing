def arr_mult(A,B):
    new = []
    for i in range(len(A)):
        newrow = []
        for j in range(len(B[0])):
            tot = 0
            for k in range(len(B)):
                tot += A[i][k] * B[k][j]
            newrow.append(tot)
        new.append(newrow)
    return new
