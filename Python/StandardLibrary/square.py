import sys

def square(n):
    return n*n

if __name__ == '__main__':
    # Print the name of the program.
    print sys.argv[0]
    # Set n equal to the number passed as the first argument.
    n = float(sys.argv[1])
    print square(n)