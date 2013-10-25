# p1 \{\(~!@#\$%\^&\*_\+\)\}\.
    # proof: python regex_bulk.py   \{\(~!@#\$%\^&\*_\+\)\}\.   {(~!@#$%^&*_+)}.
# p2 ^(Book|Mattress|Grocery) (store|supplier)$
# p3 ^[^d-w]$
    # proof: python regex_bulk.py   ^[^d-w]$    a b c d w x y z ag dg wg zg
# p4 ^[a-zA-Z_]\w\w\(\)$
    # proof: python regex_bulk.py   ^[a-zA-Z_]\w\w\(\)$    cat() Hb3() C9T() _18() ___() 3bH() fish() _() cat(mouse) ab*()
# p5 ^[a-zA-Z_]\w*\(\)$
# p6 ^[a-zA-Z_]\w*\(([a-zA-Z_]\w*(,\s*[a-zA-Z_]\w*)*)?\)$
    # proof: 
import re


pattern = re.compile(r"^[a-zA-Z_]\w*\(([a-zA-Z_]\w*(,\s*[a-zA-Z_]\w*)*)?\)$")
strings_to_match = ["compile(pattern, string)", "sleep()", "a113(_7h,        stuff)", "do_problem(error, )", "err*r(gamma)", "sleep()()"] 

def print_case(s):
    if pattern.match(s):  # This is where the work happens
        prefix = "Match:   \t"
    else:
        prefix = "No match:\t"
    print prefix, s

map(print_case, strings_to_match)
