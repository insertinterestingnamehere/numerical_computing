from sys import argv
import re


regex_pattern_string = argv[1]
strings_to_match = argv[2:]
pattern = re.compile(regex_pattern_string)

def print_case(s):
    if pattern.match(s):  # This is where the work happens
        prefix = "Match:   \t"
    else:
        prefix = "No match:\t"
    print prefix, s

map(print_case, strings_to_match)
    ### This is basically equivalent to:
        # for s in strings_to_match:
        #    print_case(s)
