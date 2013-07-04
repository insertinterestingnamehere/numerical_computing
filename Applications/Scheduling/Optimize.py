import csv
import numpy as np
from cvxopt import matrix, glpk,solvers


reader = csv.reader(open("NursesPreferences.csv",'r'), delimiter = "\t")
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
for i in range(10):
    one = []
    for j in range(700):
        if j < 70 * (i+1) and j >=70 * i:            
            one.append(1.0)
        else:
            one.append(0.0)
    constraint1.append(one)

b = np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])

reader2 = csv.reader(open("demand.csv", "r"), delimiter="\t")
demand = []
for row in reader2:
    for row1 in row:
        demand.append(-float(row1))
for i in range(700):
    demand.append(1.)
for i in range(700):
    demand.append(0.)


constraint2 = []
for k in range(14):
    array = []
    for i in range(10):
        for j in range(70):
            array.append(-float(shifts[j][k]))
    constraint2.append(array)
for k in range(14):
    array = []
    for i in range(10):
        for j in range(70):
            if level[i] < 2:
                array.append(0.)
            else:
                array.append(-float(shifts[j][k]))
    constraint2.append(array)
for i in range(700):
    array = []
    for j in range(700):
        if i == j:
            array.append(1.)
        else:
            array.append(0.)
    constraint2.append(array)
for i in range(700):
    array = []
    for j in range(700):
        if i == j:
            array.append(-1.)
        else:
            array.append(0.)
    constraint2.append(array)
    

preferences = np.array([preferences])
constraint1 = np.array(constraint1)
demand = np.array(demand)
constraint2 = np.array(constraint2)
print constraint2.shape

c = matrix(preferences.T)
G = matrix(constraint2)
h = matrix(demand)
A = matrix(constraint1)
b = matrix(b.T)
sol = glpk.ilp(c,G,h,A,b,I=set([0,1]))

solution = []
for i in range(10):
    array = []
    for j in range(70):
        k = sol[1][(i)*70+j]
        if k == 1:
            print "nurse", i, "takes shift", j
        array.append(k)
    solution.append(array)
