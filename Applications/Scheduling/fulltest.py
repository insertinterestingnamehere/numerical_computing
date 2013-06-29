import csv
import numpy as np
from cvxopt import matrix, glpk


reader = csv.reader(open("Preferences.csv",'r'), delimiter = ",")
preferences = []
level = []
for row in reader:
    level.append(float(row[0]))
    for i in range(1,71):
        preferences.append(float(row[i]))
    

reader1 = csv.reader(open("SchedulingData.csv","r"), delimiter=",")
shifts = []
for row in reader1:
    array = []
    for row1 in row:
        a = float(row1)
        array.append(a)
    shifts.append(array)


constraint1 = []
for i in range(30):
    one = []
    for j in range(2100):
        if j < 70 * (i+1) and j >=70 * i:            
            one.append(1.0)
        else:
            one.append(0.0)
    constraint1.append(one)

b = np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,\
               1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])

reader2 = csv.reader(open("Demands.csv", "r"), delimiter=",")
demand = []
for row in reader2:
    for row1 in row:
        demand.append(-float(row1))
##for i in range(2100):
##    demand.append(1.)
##for i in range(2100):
##    demand.append(0.)


constraint2 = []
for k in range(14):
    array = []
    for i in range(30):
        for j in range(70):
            array.append(-float(shifts[j][k]))
    constraint2.append(array)
for k in range(14):
    array = []
    for i in range(30):
        for j in range(70):
            if level[i] < 2:
                array.append(0.)
            else:
                array.append(-float(shifts[j][k]))
    constraint2.append(array)
for k in range(14):
    array = []
    for i in range(30):
        for j in range(70):
            if level[i] < 3:
                array.append(0.)
            else:
                array.append(-float(shifts[j][k]))
    constraint2.append(array)
##for i in range(2100):
##    array = []
##    for j in range(2100):
##        if i == j:
##            array.append(1.)
##        else:
##            array.append(0.)
##    constraint2.append(array)
##for i in range(2100):
##    array = []
##    for j in range(2100):
##        if i == j:
##            array.append(-1.)
##        else:
##            array.append(0.)
##    constraint2.append(array)
    

preferences = np.array([preferences])
constraint1 = np.array(constraint1)
demand = np.array(demand)
constraint2 = np.array(constraint2)
print constraint2.shape
print constraint1.shape
print preferences.shape
print demand.shape

c = matrix(preferences.T)
G = matrix(constraint2)
h = matrix(demand)
A = matrix(constraint1)
b = matrix(b.T)
sol = glpk.ilp(c,G,h,A,b,I=set([0,1]))
print sol[1]

solution = []
for i in range(30):
    array = []
    for j in range(70):
##        k = sol[1][(i)*70+j]
        if k == 1:
            print "nurse", i, "takes shift", j
        array.append(k)
    solution.append(array)
