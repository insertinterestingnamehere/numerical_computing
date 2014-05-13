def arr_mult(A,B):
    new = []
    # Iterate over the rows of A.
    for i in range(len(A)):
        # Create a new row to insert into the product.
        newrow = []
        # Iterate over the columns of B.
        # len(B[0]) returns the length of the first row 
        # (the number of columns).
        for j in range(len(B[0])):
            # Initializes an empty total.
            tot = 0
            # Multiply the elements of the row of A with 
            # the column of B, then sum the products.
            for k in range(len(B)):
                tot += A[i][k] * B[k][j]
            # Insert the value into the new row of the product. 
            newrow.append(tot)
        # Insert the new row into the product.
        new.append(newrow)
    return new
