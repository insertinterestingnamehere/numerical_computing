from pypermutation import PyPermutation

P = PyPermutation([[1,3,5,4,9,6,7],[2,5,4,6,8,3,1]])
print "P: ", P
P.reduce()
print "P reduced: ", P
Q = PyPermutation([[2,4,5,3,9,12],[1,3,4,2,8]])
print "Q: ", Q
Q.reduce()
print "Q reduced: ", Q
print "P * Q : ", P * Q
print "P**3 : ", P**3
raw_input("Press enter to exit")
