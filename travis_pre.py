import os
import sys

import travis_common as tc


# 100KB in bytes
MAX_FILESIZE = 102400

def getOutput(cmd):
    return os.popen(cmd).read()

def find_big_files(fatal=True):
    #load the exception file
    exceptions = set()
    with open('travis_file_exceptions', 'rU') as e:
        for L in e:
            exceptions.add(tuple(map(str.strip, L.split())))
        
    revisions = getOutput("git rev-list HEAD").split()
    violations = set()
    for r in revisions:
        tree = getOutput("git ls-tree -rlz {}".format(r)).split("\0")
        for obj in tree:
            try:
                data = obj.split()
                commit, size, name = data[2], int(data[3]), data[4]
                if (commit, name) not in exceptions and size > MAX_FILESIZE:
                    violations.add((size, name, commit))
            except (IndexError, ValueError):
                continue
    
    if violations:
        for v in sorted(violations, reverse=True):
            print "{} {} {}".format(*v)
        tc.raise_msg("Large files present", fatal=fatal, category=tc.BuildWarning)
            

if __name__ == "__main__":
    find_big_files()