import math
import cmath
import random
import timeit
import csv
import collections as col
import itertools
import sys


#Problem 1

def read_events():
    # Read in the file 'events.txt'.
    # Return a list of the events.
    events = []
    with open('events.txt','r') as f:
        for line in f:
            events.append(line.strip())
    return events

events = read_events()

#Problem 2

# Using the sys module, print the filename 'output.txt' to screen.
# 'output.txt' is an argument passed in at command line.
# In practice, this part would be better to put at the bottom of your script.
if __name__ == "__main__":
    print sys.argv[1]

#Problem 3

def read_tributes():
    males =[]
    females = []
    with open('tributes.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for male, female in csv_reader:
            males.append(male)
            females.append(female)
    return males, females

males, females = read_tributes()

#Problem 4

def sqrt_variants(n):
    # Print floating point squareroot.
    # Print complex squareroot.
    print math.sqrt(n)
    print cmath.sqrt(n)

#Problem 5

def random_list():
    # Create and return a list of 24 random floating-point numbers
    # between 1.0 and 10.0 that represent the likelihood of a tribute
    # surviving an event.
    rand = []
    for i in xrange(24):
        rand.append(random.random()*9+1)
    return rand

likelihoods = random_list()

#Problem 6

def pair_tributes(males, females):
    # The parameters 'males' and 'females' are the lists from Problem 3.
    # Create a named tuple called "Tribute".
    # Return a list of 24 named tuples, each representing a tribute.
    Tribute = col.namedtuple('Tribute', 'name, district, gender')
    tributes = []
    for d, t in enumerate(itertools.izip(males, females),1):
        tributes.append(Tribute(t[0], d, "M"))
        tributes.append(Tribute(t[1], d, "F"))
    return tributes

tributes = pair_tributes(males, females)

#Problem 7

# Initialize deque D, with 10000 elements.
# Initialize list L, with 10000 elements.

def time_func(f, args=(), kargs={}, repeat=3, number=100):
    # Wrap f into pfunc.
    pfunc = lambda: f(*args, **kargs)
    # Define an object T that times pfunc once.
    T = timeit.Timer(pfunc)

    # Time f several times, return the name of f and the minimum runtime.
    try:
        # Repeat is also a timeit module function
        t = T.repeat(repeat=repeat, number=int(number))
        runtime = min(t)/float(number)
        return runtime
    # Print an error statement if something goes wrong.
    except:
        T.print_exc()

def rotate_deque(D):
    # In this function use the deque object's rotate method.
    D.rotate(10000)

def rotate_list(L):
        length = len(L)
        for i in xrange(length):
            y = L.pop()
            L.insert(0,y)

L = range(10000)
D = col.deque(range(10000))

# Print timing for rotate_deque.
print time_func(rotate_deque, [D])
# Print timing for rotate_list.
print time_func(rotate_list, [L])



#Problem 8
def HungerSim(events, likelihoods, tributes):
    # Parameters are:
    #   events - list of events from Problem 1.
    #   likelihoods - the list of random numbers from Problem 5.
    #   tributes - the list of tributes from Problem 6.
    # Write the results of each day to the 'output.txt' file.
    k = 0
    with open('output.txt','w') as f:
        while len(tributes) > 1:
            survivors = []
            k += 1
            f.write("Day " + str(k) + "\n")
            for i in xrange(len(tributes)):
                random.shuffle(events)
                if (random.random()*9+1) > likelihoods[i]:
                    survivors.append(tributes[i])
                    f.write(str(tributes[i][0]) + " experienced " + str(events[i]) + " and survived.\n")
                else:
                    f.write(str(tributes[i][0]) + " experienced " + str(events[i]) + " and died.\n")
            f.write("End of Day " + str(k) + "\n" + " \n")
            tributes = survivors


        if len(tributes) == 0:
            print "There were no winners."
            f.write("There were no winners.")
        else:
            print "The final tribute was the girl from District " + str(tributes[0][1]) + ": " + str(tributes[0][0])
            f.write("The final tribute was the girl from District " + str(tributes[0][1]) + ": " + str(tributes[0][0]))


HungerSim(events, likelihoods, tributes)








