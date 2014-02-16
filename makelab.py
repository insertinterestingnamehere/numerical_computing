import os
import tempfile
import ConfigParser
import shutil
import subprocess
import make_util

def main(args):
    tmp_dir = tempfile.mkdtemp()
    c = ConfigParser.SafeConfigParser()
    c.readfp(open('makelab.conf', 'r'))
        
    #get current working directory
    cwd = os.getcwd()

    lab_path, lab_name = os.path.split(args.lab.strip())
    lab_path = os.path.relpath(lab_path, start=cwd)
    tmp_lab_path = os.path.join(tmp_dir, lab_path)
    
    shutil.copytree(lab_path, tmp_lab_path)
    filelist = c.get('root', 'CopyFiles', raw=True).split(',')
    template = c.get('root', 'Template', raw=True)
    template_subbed = make_util.substlab(template, lab_path, lab_name)

    for f in filelist:
        shutil.copy(f, tmp_dir)
    
    os.chdir(tmp_dir)
    with open('template.tex', 'w') as t:
        t.writelines(template_subbed)
    
    #in case we have references, we need to run multiple times.
    for f in make_util.run_seq:
        ret = f(template)
        if ret.returncode != 0:
            break

    filename = os.path.splitext(template)[0]+'.pdf'
    labname = os.path.splitext(lab_name)[0]+'.pdf'
    copydest = os.path.join(cwd, lab_path, labname)
    if ret.returncode == 0:
        shutil.copy2(filename, copydest)
        print "{} -> {}".format(os.path.join(tmp_lab_path, labname), lab_path)
    else:
        #copy template.log to copydest
        shutil.copy2("template.log", copydest)
        print "{} could not be generated".format(labname)    

if __name__ == "__main__":
    import argparse
    
    descr = """
    Makelab - An automated lab compiler.
    
    This will compile a single specified lab with references and output a pdf file in the lab's folder.
    """
    
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('lab', action='store', help="Path to lab LaTeX file")
    main(parser.parse_args())
