
#In simulating a DFA, we don't need to give the set of all states
#We just need the transition function.
#The transition function is of the form {(origin, char): transition state,...}
DFA_1 = (set("01"), {(0, '0'): 1}, 0, set([1]))

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
            
        
        