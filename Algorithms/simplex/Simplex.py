import numpy as np

def simplex(T):
   # choose the first pivot: this sets exit to
   # 1 if finished or 3 if unbounded
   i, j, exit = choosepivot(T)
   while exit == 0:
       pivot(T,i,j)
       i, j, exit = choosepivot(T)
   return i, j, exit

def createTableau(c, A, b):
    m,n = A.shape
    s = m + n + 1;    #m is number of constraints, n is number of variables 
    T = np.zeros((s,s))
    T[0, 1:n+1] = c.T
    I = np.eye(n)
    T[1:n+1, 1:n+1] = I.squeeze()
    T[n+1:s, 0] = b.squeeze()
    T[n+1:s, 1:n+1] = -A.squeeze()
    return T

def choosepivot(T):
   s = T.shape
   finished = True
   for k in range(1,s[0]):
      if T[0][k] > 0:
         finished = False
         break
   if finished == True:
      exitvalue = 1
      return 0,0,exitvalue
   objective = T[0]
   m = 0
   for k in range(1,s[1]):
      if objective[k] > m and objective[k] > 0:
         m = objective[k]
         i = k
   first = True
   for k in range(1,s[0]):
      if T[k][0] != 0 and T[k][i] != 0 and first == True:
         m = T[k][0]/T[k][i]
         j = k
         first = False
      elif T[k][0] != 0 and T[k][i] != 0:
         r = T[k][0]/T[k][i]
         if m < r and r <= 0:
            m = r       
            j = k
   exitvalue = 0
   print i,j
   return i, j, exitvalue

def pivot(T,i,j):
   U = T.copy()
   vector = U[j]
   vector[j] = -1
   m = float(vector[i]/(-1))
   row = np.divide(vector,m)
   V = np.outer(U[:,i], row)
   T = T + V
   return T

def isfeasible(T):
   return exit


def main(A,b,c):
   exitvalue = 0
   T = createTableau(c,A,b)
   print T
   i,j,exitvalue = choosepivot(T)
   iteration = 0
   while exitvalue == 0:
      T = pivot(T,i,j)
      iteration += 1
      i,j,exitvalue = choosepivot(T)
      print T
   return T[0][0], iteration, exitvalue


n = 3
A = np.zeros(shape=(n, n))
b = []
c = []
for i in range(n):
   c.append(2**(n-1-i))
   b.append(5**(i+1))
   row = []
   A[i][i] = 1
   k = 2
   for j in range(i-1,-1,-1):
      A[i][j] = 2**k
      k += 1




A = np.array(A)
b = np.array(b)
c = np.array(c)
##print A
##print b
##print c

##T = createTableau(c,A,b)
##print T
##i,j,exitvalue = choosepivot(T)
##print i,j
##print pivot(T,i,j)

print main(A,b,c)

