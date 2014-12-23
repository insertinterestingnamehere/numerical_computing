import csv
import numpy as np
from cvxopt import matrix, glpk,solvers


reader = csv.reader(open("NursesPreferences.csv",'r'), delimiter = "\t")
total = []
level = []
for row in reader:
##    array = []
    for i in range(71):
        if i == 0:
            level.append(float(row[i]))
        else:
            total.append(float(row[i]))
##    total.append(array)
    

preferences = np.array(total)

level = np.array(level)


reader1 = csv.reader(open("SchedulingData.csv","r"), delimiter=",")
data = []
for row in reader1:
    array = []
    for row1 in row:
        a = float(row1)
        array.append(a)
    data.append(array)

reader = csv.reader(open("NursesPreferences.csv","r"), delimiter="\t")
shifts = []
count = 0
for row in reader:
    if count < 4:
        count += 1
        for i in range(1,9):
            shifts.append(float(row[i]))
        
preferences = np.array([shifts])


constraint1 = []
for i in range(4):
    one = []
    for j in range(32):
        if j < 8 * (i+1) and j >=8 * i:            
            one.append(1.0)
        else:
            one.append(0.0)
    constraint1.append(one)
constraint1 = np.array(constraint1)
##    print row
b = np.array([[1.,1.,1.,1.]])
print "c", preferences.shape

print "A", constraint1.shape
print "b", b.shape

constraint2 = []

for i in range(32):
    array = []
    for j in range(32):
        if i == j:
            array.append(1.)
        else:
            array.append(0.)
    constraint2.append(array)
for i in range(32):
    array = []
    for j in range(32):
        if i == j:
            array.append(-1.)
        else:
            array.append(0.)
    constraint2.append(array)
G = np.array(constraint2)


h = []
for i in range(32):
    h.append(1.)
for i in range(32):
    h.append(0.)
h = np.array(h)
print "h", h.shape
print "G", G.shape
h = matrix(h)
G = matrix(G)
c = matrix(preferences.T)
A = matrix(constraint1)
b = matrix(b.T)
sol = glpk.ilp(c,G,h,A,b,I=set([0,1]))
solution = []
for i in range(4):
    array = []
    for j in range(8):
        array.append( sol[1][(i)*8+j])
    solution.append(array)
for row in solution:
    print row
print sol[1]
