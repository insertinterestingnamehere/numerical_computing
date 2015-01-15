import matplotlib
matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')
import numpy as np
import solutions
from matplotlib import pyplot as plt

def evolution():
    u = np.array([0,0,0,-.98])
    H = np.array([[1,0,0,0],[0,1.,0,0]])
    Q = 0.1*np.eye(4)
    R = 5000*np.eye(2)
    F = np.array([[1,0,.1,0],[0,1,0,.1],[0,0,1,0],[0,0,0,1]])
    x0 = np.array([0,0,300.,600])
    kal = solutions.KalmanFilter(F,Q,H,R,u)
    s,o = kal.evolve(x0,1250)
    
    plt.plot(s[0,:],s[1,:])
    plt.ylim(0,1.1*(s[1,:].max()))
    plt.savefig("states_evolution.pdf")
    plt.clf()
    
    plt.plot(o[0,::8],o[1,::8], 'r.')
    plt.ylim(0,1.1*(s[1,:].max()))
    plt.savefig("obs_evolution.pdf")
    plt.clf()

def norms():
    u = np.array([0,0,0,-.98])
    H = np.array([[1,0,0,0],[0,1.,0,0]])
    Q = 0.1*np.eye(4)
    R = 5000*np.eye(2)
    F = np.array([[1,0,.1,0],[0,1,0,.1],[0,0,1,0],[0,0,0,1]])
    x0 = np.array([0,0,300.,600])
    kal = solutions.KalmanFilter(F,Q,H,R,u)
    s,o = kal.evolve(x0,1250)
    
    ave_vel = np.diff(o[:,200:210], axis=1).mean(axis=1)/.1
    x = np.zeros(4)
    x[:2] = o[:,200]
    x[2:] = ave_vel
    P = 10**6*Q
    estimates, norms = kal.estimate(x,P,o[:,201:801], return_norms=True)
    
    plt.plot(norms)
    plt.savefig("norms.pdf")
    plt.clf()

def estimates():
    u = np.array([0,0,0,-.98])
    H = np.array([[1,0,0,0],[0,1.,0,0]])
    Q = 0.1*np.eye(4)
    R = 5000*np.eye(2)
    F = np.array([[1,0,.1,0],[0,1,0,.1],[0,0,1,0],[0,0,0,1]])
    x0 = np.array([0,0,300.,600])
    kal = solutions.KalmanFilter(F,Q,H,R,u)
    s,o = kal.evolve(x0,1250)
    
    ave_vel = np.diff(o[:,200:210], axis=1).mean(axis=1)/.1
    x = np.zeros(4)
    x[:2] = o[:,200]
    x[2:] = ave_vel
    P = 10**6*Q
    estimates = kal.estimate(x,P,o[:,201:801])
    
    plt.plot(norms)
    plt.savefig("norms.pdf")
    plt.clf()
    
    plt.plot(s[0,:][np.where(s[1,:]>=0)], s[1,:][np.where(s[1,:]>=0)])
    plt.plot(o[0,201:801], o[1,201:801], 'r.')
    plt.plot(estimates[0,:],estimates[1,:], 'g')
    plt.savefig("estimate_macro.pdf")
    plt.clf()
    
    S = 250
    E = S+50
    plt.plot(s[0,S:E], s[1,S:E])
    plt.plot(o[0,S:E], o[1,S:E], 'r.')
    plt.plot(estimates[0,S-201:E-201],estimates[1,S-201:E-201], 'g')
    plt.savefig("estimate_micro.pdf")
    plt.clf()

def impact():
    u = np.array([0,0,0,-.98])
    H = np.array([[1,0,0,0],[0,1.,0,0]])
    Q = 0.1*np.eye(4)
    R = 5000*np.eye(2)
    F = np.array([[1,0,.1,0],[0,1,0,.1],[0,0,1,0],[0,0,0,1]])
    x0 = np.array([0,0,300.,600])
    kal = solutions.KalmanFilter(F,Q,H,R,u)
    s,o = kal.evolve(x0,1250)
    
    ave_vel = np.diff(o[:,200:210], axis=1).mean(axis=1)/.1
    x = np.zeros(4)
    x[:2] = o[:,200]
    x[2:] = ave_vel
    P = 10**6*Q
    estimates = kal.estimate(x,P,o[:,201:801])
    predicted = kal.predict(estimates[:,-1],450)
    
    plt.plot(s[0,:], s[1,:])
    plt.plot(predicted[0,:], predicted[1,:], 'y')
    plt.ylim(0)
    plt.savefig("impact_macro.pdf")
    plt.clf()
    
    x1 = s[0,:][np.where(s[1,:]>=0)][-1]
    x2 = predicted[0,:][np.where(predicted[1,:]>=0)][-1]
    plt.plot(s[0,:], s[1,:])
    plt.plot(predicted[0,:], predicted[1,:], 'y')
    plt.ylim(0,100)
    plt.xlim(min(x1,x2)-50, max(x1,x2)+50)
    plt.savefig("impact_micro.pdf")
    plt.clf()

def origin():
    u = np.array([0,0,0,-.98])
    H = np.array([[1,0,0,0],[0,1.,0,0]])
    Q = 0.1*np.eye(4)
    R = 5000*np.eye(2)
    F = np.array([[1,0,.1,0],[0,1,0,.1],[0,0,1,0],[0,0,0,1]])
    x0 = np.array([0,0,300.,600])
    kal = solutions.KalmanFilter(F,Q,H,R,u)
    s,o = kal.evolve(x0,1250)
    
    ave_vel = np.diff(o[:,200:210], axis=1).mean(axis=1)/.1
    x = np.zeros(4)
    x[:2] = o[:,200]
    x[2:] = ave_vel
    P = 10**6*Q
    estimates = kal.estimate(x,P,o[:,201:801])
    rewound = kal.rewind(estimates[:,49],300)

    plt.plot(s[0,:],s[1,:])
    plt.plot(rewound[0,:],rewound[1,:])
    plt.ylim(0)
    plt.savefig("origin_macro.pdf")
    plt.clf()
    
    x1 = s[0,:][np.where(s[1,:]>=0)][0]
    x2 = rewound[0,:][np.where(rewound[1,:]>=0)][0]
    plt.plot(s[0,:],s[1,:])
    plt.plot(rewound[0,:],rewound[1,:])
    plt.ylim(0,100)
    plt.xlim(min(x1,x2)-50, max(x1,x2)+50)
    plt.savefig("origin_micro.pdf")
    plt.clf()
 
if __name__ == "__main__":
    norms()


