import re


def get_pattern():
    pattern_string = raw_input("Enter a regexp>")
    return pattern_string if len(pattern_string) else None

def get_string():
    string = raw_input("  Enter a string>")
    return string if len(string) else None

def print_case(pattern, s):
    if re.match(pattern, s):  # This is where the work happens
        print "    {pattern} \t matches:       \t {s}".format(**locals())
    else:
        print "    {pattern} \t doesn't match: \t {s}".format(**locals())


def main():
    pattern = get_pattern()
    while pattern:
        string = get_string()
        if string:
            print_case(pattern, string)
        else:
            pattern = get_pattern()

main()
