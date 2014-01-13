# This document is correct, but not too robust
# TODO add latex \label s

# p1 \^\{\(!%\.\*_\)\}\&
    # proof: >>> bool(re.match(r"\^\{\(!%\.\*_\)\}\&", "^{(!%.*_)}&"))
# p2 The string "two fish" and any string starting with "one"
# p3 ^(Book|Mattress|Grocery) (store|supplier)$
# p4 ^[^d-w]$
    # proof: python regex_bulk.py   ^[^d-w]$    a b c d w x y z ag dg wg zg
# p5 is_valid = lambda string: len(string)==5 and bool(re.match(r"^[a-zA-Z_]\w\w\w\w$", string))
# p6 see match_function_definition.py
# p7 see match_function_definition.py
# p8 see match_function_definition.py
