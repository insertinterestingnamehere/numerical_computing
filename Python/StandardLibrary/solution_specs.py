import math
import cmath
import random
import timeit
import csv
import collections as col
import itertools
import sys

# Problem 1

def read_events():
    # Read in the file 'events.txt'.
    # Return a list of the events.
    pass

# Problem 2

# Using the sys module, print the filename 'output.txt' to screen.
# 'output.txt' is an argument passed in at command line.
# In practice, this part would be better to put at the bottom of your script.

# Problem 3

def read_tributes():
    # Using the csv module, read in the male and female tributes.
    # Return two lists, one list containing the male tributes and
    # one list containing the female tributes.
    pass

# Problem 4

def sqrt_variants(n):
    # Print floating point squareroot.
    # Print complex squareroot.
    pass

# Problem 5

def random_list():
    # Create and return a list of 24 random floating-point numbers
    # between 1.0 and 10.0 that represent the likelihood of a tribute
    # surviving an event.
    pass

# Problem 6
def pair_tributes(males, females):
    # The parameters 'males' and 'females' are the lists from Problem 3.
    # Create a named tuple called "Tribute".
    # Return a list of 24 named tuples, each representing a tribute.
    pass

# Problem 7

# Initialize deque D, with 10000 elements.
# Initialize list L, with 10000 elements.

def rotate_deque(D):
    # In this function use the deque object's rotate method.
    pass
def rotate_list(L):
    pass
# Print timing for rotate_deque.
# Print timing for rotate_list.

# Problem 8
def HungerSim(events, likelihoods, tributes):
    # Parameters are
    #   events - list of events from Problem 1.
    #   likelihoods - the list of random numbers from Problem 5.
    #   tributes - the list of tributes from Problem 6.
    # Write the results of each day to the 'output.txt' file.
    pass

# Example of possible file output for the last two days:
'''
Day 3
Silver Herriot experienced Tracker Jackers and survived.
Hammil Odinshoot experienced Wild Deer and died.
End of Day 3

Day 4
The final tribute was the girl from District 4: Silver Herriot

'''

# Example of possible final output to console:
'''
The final tribute was the girl from District 4: Silver Herriot
'''


