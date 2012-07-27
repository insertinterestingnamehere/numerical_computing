a = float(raw_input("Enter a: "))
b = float(raw_input("Enter b: "))

if a == 0:
    print b

while b != 0:
    if a > b:
        a = a - b
    else:
        b = b - a

print a