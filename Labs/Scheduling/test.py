import csv
from random import random, randint

writer = csv.writer(open("NPreferences.csv", "w"), delimiter=",")
for i in range(30):
    array = []
    r = randint(1,4)
    array.append(r)
    for j in range(70):
        r = randint(0,50)
        array.append(r)
    writer.writerow(array)
    
        
