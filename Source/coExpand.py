import scipy as sp

def coExpand(M):
    #if not M.shape[0] == M.shape[1]:
    #    raise Exception, "Matrix must be square"

    M = sp.atleast_2d(M)
    if M.size == 1:
        print M.item(0,0)
        return M.item(0,0)
    else:
        print M
        first = M.item(0,0)*coExpand(M[1:, 1:])
        last = M.item(0,-1)*((-1)**(len(M)-1))*coExpand(M[1:, :-1])

        #det = 0
        #for i in range(1, len(M)):
        #    det += M[0,i]*((-1)**(i+1)*coExpand(sp.r_[M[1,:][:i-1], M[1,:][i+1:]]))
        #return first+last

        #~ det = 0
        #~ for i in range(len(M)):
            #~ det += M[0,i]*((-1)**(i+1)*coeffExpand(sp.r_[M[1,:][:i

#A = sp.random.randint(10, size=(3,3))
#coExpand(A)
