
#In simulating a DFA, we don't need to give the set of all states
#We just need the transition function.
#The transition function is of the form {(origin, char): transition state,...}
reprDFA_a = (set('ab'),
             {(0, 'a'): 0,
              (0, 'b'): 1,
              (1, 'b'): 0,
              (1, 'a'): 1},
             0,
             {1})
reprDFA_b = (set('01'),
             {(0, '1'): 0,
              (0, '0'): 1,
              (1, '1'): 1,
              (1, '0'): 2,
              (2, '1'): 2,
              (2, '0'): 3,
              (3, '0'): 3,
              (3, '1'): 3}
             0,
             {2})
reprDFA_c = (set('01'),
             {(0, '0'): 2,
              (0, '1'): 1,
              (1, '0'): 2,
              (1, '1'): 3,
              (3, '0'): 2,
              (3, '1'): 4,
              (4, '0'): 2,
              (4, '1'): 2,
              (2, '0'): 2,
              (2, '1'): 2},
             0,
             {1, 2})

equiv_regex_a = "a*(ba*b)*"
equiv_regex_b = "1*01*01*(0+1)*"
equiv_regex_c = ""

def dfa_sim(input_str, machine):
    #unpack the machine
    input_len = len(input_str) - 1
    alphabet, transitions, curr, accept = machine
    for i, c in enumerate(input_str):
        #match the character
        if c in alphabet:
            if (curr, c) in transitions:
                curr = transitions[(curr, c)]
            else:
                break
        else:
            raise ValueError("Invalid symbol")
        if i == input_len and curr in accept:
            return True
    return False

def email_validator(email):
    pattern = r'([\w\d\.-]+)@([\w\d\.-]+)\.([a-z]{2,4})'
    
    mo = re.match(pattern, email)
    if mo:
        s = slice(*mo.span(2))
        xs = 'x'*len(email[s])
    else:
        raise ValueError("invalid email")
    
    # replace domain with x's
    return ''.join([mo.group(1), 
                    '@', xs, '.',
                    mo.group(3)])