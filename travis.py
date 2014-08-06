from os.path import isfile as i
assert i('Vol1.pdf') and i('Vol2.pdf') and i('Vol3.pdf') and i('Vol4.pdf')
