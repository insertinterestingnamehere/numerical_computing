import os
import subprocess

def substlab(template, lab_path, lab_file):
    """Read template and substitute labs"""
    with open(template, 'r') as temp:
        t = temp.readlines()

    i = t.index('%TEMPLATE_INSERT\n')
    t.insert(i+1, '\\subimport{{{0}/}}{{{1}}}\n'.format(lab_path.replace('\\', '/'), lab_file))
    return t

def getDirs(root):
    return set([x for x in os.listdir(root) if os.path.isdir(x)])
    
def run_latex(template, msg=True):
    if msg:
        print "Running XeLaTeX on {}".format(template)
        
    ret = subprocess.Popen(['xelatex', '-halt-on-error', template],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout = ret.communicate()[0]
    return ret
    
def run_biber(template, msg=True):
    if msg:
        print "Running Biber on {}".format(template)
        
    _t = os.path.splitext(template)[0]
    ret = subprocess.Popen(['biber', _t])
    stdout = ret.communicate()[0]
    return ret

run_seq = [run_latex,
           run_biber,
           run_latex,
           run_latex]
