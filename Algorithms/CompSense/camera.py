import numpy as np

class Camera:
    def __init__(self, faces, verts, C):
        Fs = self._buildfaces(faces, verts)
        l = len(Fs)
        self.F = np.ones((l, 4, 3))
        self.F[:, :-1, :] = Fs
        self.C = C
        self.b = None
        self.M = None
        
    def _buildfaces(self, facetM, verM):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.zeros(3)
        l = len(facetM)
        F = np.empty((l, 3, 3))
        for n in xrange(l):
            x[0] = verM[facetM[n, 0]-1, 0]
            y[0] = verM[facetM[n, 0]-1, 1]
            z[0] = verM[facetM[n, 0]-1, 2]
            x[1] = verM[facetM[n, 1]-1, 0]
            y[1] = verM[facetM[n, 1]-1, 1]
            z[1] = verM[facetM[n, 1]-1, 2]
            x[2] = verM[facetM[n, 2]-1, 0]
            y[2] = verM[facetM[n, 2]-1, 1]
            z[2] = verM[facetM[n, 2]-1, 2]
            verts = np.array([x, y, z])
            F[n] = verts
        return F

    def _Transform(self, theta, psi, r=3):
        c = np.array([r*np.sin(psi)*np.cos(theta), r*np.sin(psi)*np.sin(theta), r*np.cos(psi)])
        cnorm = np.linalg.norm(c)
        t = np.arccos(-c[2]/cnorm)
        wstar = np.array([c[1]/cnorm, -c[0]/cnorm, 0])
        w = wstar/np.linalg.norm(wstar)
        what = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        R = np.eye(3)+np.sin(t)*what+(1-np.cos(t))*np.linalg.matrix_power(what, 2)
        P = np.zeros((4, 4))
        P[:3, :3] = R.T
        P[:3, -1] = np.dot(-R.T ,c)
        P[-1, -1] = 1
        return P, c
    
    def _SPC(self, theta, psi, r=3):
        P, c = self._Transform(theta, psi, r)
        def Pc(y,f=.5):
            return np.array([y[0]/(f*y[2]), y[1]/(f*y[2])])
        l = len(self.F)
        A = np.empty(l)
        for i in xrange(l):
            v1 = Pc(np.dot(P, self.F[i, :, 1]))-Pc(np.dot(P, self.F[i, :, 0]))
            v2 = Pc(np.dot(P, self.F[i, :, 2]))-Pc(np.dot(P, self.F[i, :, 0]))
            A[i] = .5*np.abs(v1[0]*v2[1]-v1[1]*v2[0])
        centers = np.mean(self.F[:, :-1, :], axis=-1)
        e = np.sqrt(np.sum((c-centers)**2, axis=1))<np.sqrt(10.)
        M = e*A
        b = np.empty(3)
        b = np.dot(M, self.C)
        return M, b
    def add_pic(self, theta, psi, r=3):
        M, b = self._SPC(theta, psi, r)
        if self.b is None:
            self.b = np.array([b])
            self.M = np.array([M])
        else:
            self.b = np.concatenate((self.b, np.array([b])), axis=0)
            self.M = np.concatenate((self.M, np.array([M])), axis=0)
    def add_lots_pic(self, n, r=3):
        self.b = np.empty((n, 3))
        self.M = np.empty((n, 1012))
        for i in xrange(n):
            theta = np.random.rand()*np.pi
            psi = np.random.rand()*2*np.pi
            M, b = self._SPC(theta, psi, r)
            self.b[i] = b
            self.M[i] = M
    def returnData(self):
        return self.M, self.b
    def clear(self):
        self.b = None
        self.M = None