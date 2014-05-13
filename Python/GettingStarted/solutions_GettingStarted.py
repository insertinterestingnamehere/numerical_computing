# Problem 1
'''
1. Integer Division returns the floor.
2. Imaginary numbers are written with a suffix of j or J.
	Complex numbers can be created with the complex(real, imag) function.
	To extract just the real part use .real
	To extract just the imaginary part use .imag
3. float(x) where x is the integer.
4. //
5. A string is immutable because its content cannot be changed. 
6. string[::2] returns every other letter of the string. 
	string[27:0:-1] returns the string in reverse - without the first character.
	The entire string in reverse can be accessed by string[::-1]
'''

# Problem 2	
'''
1. Mutable objects can be changed in place after creation. The value stored in memory is changed.
 Immutable objects cannot be modified after creation. 
2.  a[4] = "yoga"
	a[:] is a copy of the entire list
	a[:] = [] clears the list (del a[:] is also an option)
	len(a) returns the list size. 
	a[0], a[1] = "Peter Pan", "camelbak"
	a.append("Jonathan, my pet fish")
3. set() (must be used to create an empty set) and {}
4. dict() and {} (must be used to create an empty dictionary)
5. sq = {x: x**2 for x in range(2,11,2)}
6. del(dict[key])
7. dict.keys(); dict.values()
	
'''

# Problem 3
'''	
1. 	The print statement writes the value of the expression(s) it's given 
	to the standard output. The return statement allows a function to specify 
	a return value to be passed back to the calling function. 
2. Grocery List cannot have a space. It is also important NOT to call your list "list". 
Doing so shadows Python's built in list constructor. 
	for loop and if statement require a colon and nested indentation
	i%2 == 0. Not an assignment. 
	Grocery List[i]. Needs brackets.
	
	
'''

# Problem 4
'''
1. my_list = []
my_list = [i for i in xrange(5)]
my_list[3] = float(my_list[3])
del my_list[2]
my_list.sort(reverse=True)

2. setA, setB = set(), set()
setA = {i for i in range(5)}
setB = {str(i) for i in range(5)}
union = set.union(setA, setB)


'''