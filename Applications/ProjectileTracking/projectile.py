import scipy as sp
import pickle
import kalman
import matplotlib.pyplot as plt
from scipy import linalg as la

### Part 1 ###
# Consider a projectile object starting from (0,0) with initial velocity (300,600) m/s.
# Suppose we have a gravitational acceleration in y of -9.8 m/s. 
# Suppose Q = 0.1*I, and R = 500*I, and delta t = 0.1.
# Evolve the system forward 1200 steps and keep the states and observations.

Fk = sp.array([[1.,0.,.1,0.],[0.,1.,0.,.1],[0.,0.,1.,0.],[0.,0.,0.,1.]])
Q = sp.eye(4)*.1
U = sp.array([0.,0.,0.,-.98])
H = sp.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
R = sp.eye(2)*500

x_initial = sp.array([0.,0.,300.,600.])

states, observations = kalman.generation(Fk,Q,U,H,R,x_initial,1250)

### Part 2 ###
# Suppose we are only able to see the observations from iteration 200 to 800.
# Plot these observations as red points and the entire projectile path as a blue curve.

plt.plot(observations[0,200:800],observations[1,200:800],'r.')
temp = sp.array([x > -50 for x in states[1,:]])
plt.plot(states[0,temp],states[1,temp],'b')

### Part 3 ###
# Supposing we only have the given observations, estimate the state of the system at 
# iteration 200, using the average of the measured velocities from iteration 200 to 210.
# Estimate the position of the projectile given this initial state estimate, using P = 10^6 * Q.
# Add to the plot the estimated path of the projectile as a green curve.

vel = sp.array([sp.mean(sp.diff(observations[0,200:210])/.1),sp.mean(sp.diff(observations[1,200:210])/.1)])
x_est_initial = sp.concatenate([observations[:,200],vel])

estimation = kalman.kalmanFilter(Fk,Q,U,H,R,x_est_initial,Q*(10**6),observations[:,200:800])

plt.plot(estimation[0,:],estimation[1,:],'g')

### Part 4 ###
# Given the final state estimate at iteration 800, iterate forward predictively to find the 
# projectile's point of impact. Plot this with a yellow curve.

prediction = kalman.predict(Fk,U,estimation[:,599],500)
temp = sp.array([x > -50 for x in prediction[1,:]])
plt.plot(prediction[0,temp],prediction[1,temp],'y')

### Part 4 ###
# Given the state estimate at iteration 250, rewind the system to identify the
# projectile's point of origin. Plot this with a cyan curve, and display the results.

rewound = kalman.rewind(Fk,U,estimation[:,50],300)
temp = sp.array([x > -50 for x in rewound[1,:]])
plt.plot(rewound[0,temp],rewound[1,temp],'c')
plt.ylim([0,20000])
plt.show()
