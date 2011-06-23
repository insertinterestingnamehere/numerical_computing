def Problem8(n):
    """Verify the numerical accuracy of linalg.lstsq vs la.inv"""
    from scipy.linalg import lstsq, inv, norm
    from scipy import dot, rand, allclose

    A = rand(n,n)
    b = rand(n,1)

    inv_method=dot(inv(A), b)
    lstsq_method=lstsq(A,b)[0]

    #check the accuracy
    return norm(inv_method-lstsq_method)