N=100
Wbig=sp.zeros((N,N))
i=0
W=sp.linspace(0,1,N)
for x in W:
    Wbig[:,i]=W-x
    i=i+1
u=(Wbig<0)*-Wbig+Wbig
u=sp.sqrt(u)
u=(Wbig<0)*-10**10+u

I = sp.sparse.identity(N)

policy=sp.zeros((N))
b=.9
d=1
while d>10**-9:
    psi_ind=sp.array(policy,dtype='int')
    rows = sp.arange(0,N) 
    columns = psi_ind.copy() 
    data = sp.ones(N)
    Q = sp.sparse.coo_matrix((data,(rows,columns)),shape = (N,N)) 
    Q = Q.tocsr()
    V = spsolve(I-b*Q,sp.sqrt(W-W[psi_ind]))
    #W2=abs(W-policy[:,i])
    #W3=(W2<0)*-W2+W2
    #W1=sp.sqrt(W3)
    #V=sp.zeros((1,100))
    #for t in range(125):
    #    V=V+b**t*W1
    Value=sp.zeros((N,N))
    for j in range(N):
        Value[j,:]=V.T*b
    total=u+Value
    temp1= total.argmax(1)
    d=la.norm(policy-temp1)
    policy=temp1.copy()

plt.plot(W,policy/99.0)
plt.show()

N=100
Wbig=sp.zeros((N,N))
i=0
W=sp.linspace(0,1,N)
for x in W:
    Wbig[:,i]=W-x
    i=i+1
u=(Wbig<0)*-Wbig+Wbig
u=sp.sqrt(u)
u=(Wbig<0)*-10**10+u

I = sp.sparse.identity(N)
V=sp.zeros(N)
policy=sp.zeros((N))
b=.9
d=1
while d>10**-9:
    psi_ind=sp.array(policy,dtype='int')
    rows = sp.arange(0,N) 
    columns = psi_ind.copy() 
    data = sp.ones(N)
    Q = sp.sparse.coo_matrix((data,(rows,columns)),shape = (N,N)) 
    Q = Q.tocsr()
    for t in range(15):
        V = (b*Q.dot(V))+sp.sqrt(W-W[psi_ind])
    Value=sp.zeros((N,N))
    for j in range(N):
        Value[j,:]=V*b
    total=u+Value
    temp1= total.argmax(1)
    d=la.norm(policy-temp1)
    policy=temp1.copy()

plt.plot(W,policy/99.0)
plt.show()