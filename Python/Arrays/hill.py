import itertools
import string
import numpy as np
from scipy import linalg
from fractions import gcd

def egcd(a, b):
    '''
    Extended Euclidean algorithm
    Returns (b, x, y) such that mx + ny = b
    Source: http://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    '''
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q,r = b//a,b%a; m,n = x-u*q,y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y

def modinv(a, m):
    '''
    Find the modular inverse.
    Source: http://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    '''
    g, x, y = egcd(a, m)
    if g != 1:
        return None  # modular inverse does not exist
    else:
        return x % m

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def blockize(msg, n):
    lut = {a:i for i ,a in enumerate(string.lowercase)}
    msg = "".join(msg.lower().split())
    return list(map(np.array, grouper(map(lut.__getitem__, msg), n, fillvalue=lut['x'])))

def inv_mat(n):
    tries = 0
    while True:
        a = np.random.randint(1000, size=(n, n)) % 26
        d = round(linalg.det(a))
        
        if gcd(int(d), 26) == 1:
            break
        tries += 1
            
    return a, d

def encode(msg, k):
    ciphertext = []
    n = k.shape[0]
    ilut = {i:a for i, a in enumerate(string.lowercase)}
    for i in blockize(msg, n):
        s = i.dot(k) % 26
        ciphertext.append("".join(map(ilut.__getitem__, s)))
    
    return "".join(ciphertext)


def inv_key(key):
    d = round(linalg.det(key))
    inv_d = modinv(int(d), 26)
    ik = np.round(d*linalg.inv(key))
    return (ik*inv_d) % 26
    
def decode(msg, k):
    ik = inv_key(k)
    n = ik.shape[0]
    plaintext = []
    ilut = {i:a for i, a in enumerate(string.lowercase)}
    for i in blockize(msg, n):
        s = i.dot(ik) % 26
        plaintext.append("".join(map(ilut.__getitem__, s)))
        
    return "".join(plaintext)

    
    