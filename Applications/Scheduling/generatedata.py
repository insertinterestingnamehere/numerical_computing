import csv
from random import randint
import numpy as np

##
##w = csv.writer(open("NursesPreference.csv", "wb"),quoting=csv.QUOTE_NONE)
##level = [1,2,2,1,1,1,2,1,1,1]
##for i in range(0,10):
##    nurse = []
##    nurse.append(level[i])
##    for i in range(70):
##        r = randint(0,25)
##        nurse.append(r)
##    
##    w.writerow(nurse)        
##     
##               
    
reader = csv.reader(open("NursesPreferec.csv",'rb'))
c = []
for row in reader:
    array = []
    for row1 in row:
##        print row1, "3"
        a = int(row1)
##        print 
        array.append(a)
    c.append(array)

print c.size
