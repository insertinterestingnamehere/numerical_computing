import os
import tempfile
import ConfigParser
import shutil
import subprocess

class ConfParser(ConfigParser.SafeConfigParser):

    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d

def substlab(template, lab_path, lab_file):
    """Read template and substitute labs"""
    with open(template, 'r') as temp:
        t = temp.readlines()

    i = t.index('%TEMPLATE_INSERT\n')

    t.insert(i+1, '\\subimport{{{0}{2}}}{{{1}}}\n'.format(lab_path, lab_file, os.sep))
    return t


def getDirs(root):
    return set([x for x in os.listdir(root) if os.path.isdir(x)])
    
def main(args):
    tmp_dir = tempfile.mkdtemp()
    c = ConfParser()
    with open('makelab.conf', 'r') as conf:
        c.readfp(conf)
        config = c.as_dict()
        
    #get current working directory
    cwd = os.getcwd()

    lab_path, lab_name = os.path.split(args.lab)
    lab_path = os.path.relpath(lab_path, start=cwd)
    tmp_lab_path = os.path.join(tmp_dir, lab_path)
    
    shutil.copytree(lab_path, tmp_lab_path)
    filelist = config['root']['copyfiles'].split(',')
    template = config['root']['template']
    template_subbed = substlab(template, lab_path, lab_name)

    for f in filelist:
        shutil.copy(f, tmp_dir)
    
    os.chdir(tmp_dir)
    with open('template.tex', 'w') as t:
        t.writelines(template_subbed)
    ret = subprocess.Popen(['xelatex', '-halt-on-error', template],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    filename = os.path.splitext(template)[0]+'.pdf'
    labname = os.path.splitext(lab_name)[0]+'.pdf'
    stdout = ret.communicate()[0]

    if ret.returncode == 0:
        shutil.copy2(filename, os.path.join(cwd, lab_path, labname))
        print "{} -> {}".format(os.path.join(tmp_lab_path, labname), lab_path)
    else:
        print stdout
        print "{} could not be generated".format(labname)    


if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser(description="makelab")
    parser.add_argument('lab', action='store')
    main(parser.parse_args())


#makelab ComplexIntegration/ComplexIntegration1.tex
#makelab ComplexIntegration1.tex
