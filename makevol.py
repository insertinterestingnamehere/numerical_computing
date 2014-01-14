import make_util


def main(args):
    for f in make_util.run_seq:
        ret = f(args.vol)
        if ret.returncode != 0:
            break
        
    if ret.returncode == 0:
        print "Success!"
    else:
        #copy template.log to copydest
        print "{} could not be generated".format(args.vol)    
        
if __name__ == "__main__":
    import argparse
    
    descr = """
    Makelab - An automated lab compiler.
    
    This will compile a selected volume of labs with references and output a pdf file in the main directory.
    """
    
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('vol', action='store', help="Volume to generate with references")
    main(parser.parse_args())
