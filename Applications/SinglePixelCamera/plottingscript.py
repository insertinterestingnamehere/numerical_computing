import matplotlib . pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
import numpy as np
import matplotlib . pyplot as plt
from matplotlib.collections import PolyCollection
def visualize2d(verts,C):
    fig, ax = plt.subplots()
    coll = PolyCollection(verts, facecolors=C)
    ax.add_collection(coll)
    ax.autoscale_view()
    plt.savefig('project')
    plt.show()
def visualize3d(F,C):
    fig = plt.figure()
    ax = Axes3D(fig)
    l=len(F)
    for n in xrange(l):
        verts = [zip(F[n,0]*.5+.5,F[n,1]*.5+.5,F[n,2]*.5+.5)]
        tri=Poly3DCollection(verts)
        tri.set_color(colors.rgb2hex(C[n]))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    plt.show()
def Project(F,C,omega,psi,r=3):
    P,c=Transform(omega,psi,r)
    def Pc(y,f=.5):
        return np.array([y[0]/(f*y[2]),y[1]/(f*y[2])])
    l=len(F)
    Q=np.empty((l,3,2))
    for i in xrange(l):
        Q[i]=Pc(np.dot(P,F[i,:,:])).T
    centers=np.mean(F[:,:-1,:],axis=-1)
    '''
    newcen=np.ones((l,4))
    newcen[:,:3]=centers
    dist= np.dot(P,newcen.T)[2]
    e=dist<3
    '''
    e=np.sqrt(np.sum((c-centers)**2,axis=1))<np.sqrt(10.)-.2
    return Q[e],e
test=np.array([[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,-1]],[[1,0,0],[0,-1,0],[0,0,-1]],
               [[1,0,0],[0,-1,0],[0,0,1]],[[-1,0,0],[0,1,0],[0,0,1]],[[-1,0,0],[0,-1,0],[0,0,1]]
               ,[[-1,0,0],[0,1,0],[0,0,-1]],[[-1,0,0],[0,-1,0],[0,0,-1]]])
testC=np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,1],[0,1,1],[1,1,0],[1,1,1],[0,0,0]])
visualize3d(test,testC)
Q,e=Project(Ft,testC,5*np.pi/8,np.pi/2)
visualize2d(Q,testC[e])