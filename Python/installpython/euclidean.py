a = int(raw_input("Enter a: "))
b = int(raw_input("Enter b: "))

while b != 0:
    tmp = b
    b = a % b
    a = tmp
    
print "The gcd is ", a
