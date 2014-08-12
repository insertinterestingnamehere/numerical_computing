import os
import travis_common as tc

def all_present(fatal=True):
    try:
        assert os.path.isfile('Vol1.pdf')
        assert os.path.isfile('Vol2.pdf')
        assert os.path.isfile('Vol3.pdf')
        assert os.path.isfile('Vol4.pdf')
    except AssertionError as e:
        raise BuildError(e)
    
if __name__ == "__main":
    all_present()