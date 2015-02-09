import string
import random
import math
import numpy as np
from itertools import izip_longest
from collections import namedtuple
from fractions import gcd

PublicKey = namedtuple("PublicKey", ['e', 'n'])
PrivateKey = namedtuple("PrivateKey", ['d', 'n', 'phi_n'])

def genkeys(bits, e):
    p = nextprime(random.getrandbits(bits//2))
    q = nextprime(random.getrandbits(bits//2))
    n = p * q
    pn = (p-1)*(q-1)
    
    e %= pn
    while gcd(e, pn) != 1 or e < 2:
        e = (e + 1) % pn
    d = modinv(e, pn)
    return PublicKey(e, n), PrivateKey(d, n, pn)

def mod_nai(b, e, m):
    c = b
    loops = 0
    largest = 0
    for x in xrange(1, e):
        c = c * b
        largest = largest if c < largest else c
        c %= m
        loops += 1
    return c, loops, largest

def mod_sch(b, e, m):
    result = 1
    b %= m
    loops = 0
    while e > 0:
        if e % 2 == 1:
            result = (result * b) % m
        e >>= 1
        b = (b * b) % m
        loops += 1
    return result, loops

def egcd(a, b):
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    gcd = b
    return gcd, x, y

def modinv(a, m):
    gcd, x, y = egcd(a, m)
    if gcd != 1:
        return None  # modular inverse does not exist
    else:
        return x % m

def fermat(n):
    prime = set()
    for a in xrange(2, n):
        if pow(a, n-1, n) == 1:
            prime.add(a)
    return prime

def witnessmap(n):
    r = (n-2)**.5
    wm = np.zeros((r+1,r+1))
    f = fermat(n)
    for i, _ in np.ndenumerate(wm.flat):
        if i[0]+2 < n:
            wm.flat[i] = 228 if i[0]+2 in f else 128
    return wm

def isprime(n, confidence=.99):
    err = 1 - confidence
    t = 0
    while 1./pow(2, t) > err:
        a = random.randrange(2, n)
        if pow(a, n-1, n) != 1:
            return False
        t += 1
    return True

def nextprime(n, confidence=.99):
    while not isprime(n + 1):
        n += 1
    return n + 1
        
def makeint(msg):
    newmsg = []
    for c in msg:
        newmsg.append(str2int[c])
    return ''.join(newmsg)

def makestr(msg):
    newmsg = []
    for c in msg[::2]:
        newmsg.append(int2str[c])
    return ''.join(newmsg)

def s2i(msg):
    #convert message to binary
    if not isinstance(msg, bytearray):
        msg = bytearray(msg)
    binmsg = []
    for c in msg:
        binmsg.append(bin(c)[2:].zfill(8))
    return int(''.join(binmsg), 2)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def i2s(msg):
    binmsg = bin(msg)[2:]
    binmsg = "0"*(8-(len(binmsg)%8)) + binmsg
    msg = bytearray()
    for block in grouper(binmsg, 8):
        msg.append(int(''.join(block), 2))
    return msg

def encrypt(key, message):
    m = s2i(message)
    if m.bit_length() > key.n.bit_length():
        raise ValueError("Key must be at least {} bits to encrypt this message.".format(m.bit_length()))
    
    c = pow(m, key.e, key.n)
    return i2s(c)

def decrypt(key, message):
    c = s2i(message)
    
    m = pow(c, key.d, key.n)
    return i2s(m)