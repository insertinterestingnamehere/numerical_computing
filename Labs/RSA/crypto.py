from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP as oaep
from random import shuffle
from Crypto.Signature import PKCS1_PSS as pss
from Crypto.Hash import SHA as sha

def make_key(nbits):
    # generate a random key of nbits
    return RSA.generate(nbits)

def getpublickey(key):
    # extract the public part of a key
    pkey = key.publickey()
    share = pkey.exportKey()
    return share

def read_group_keys(keys):
    # read an RSA key and convert it to a key object
    group_keys = map(RSA.importKey, keys)
    return group_keys

def generate_keys(nkeys, nbits=2048):
    # generate private/public keypairs as RSA key objects
    private_keys = [make_key(nbits) for x in xrange(nkeys)]
    public_keys = read_group_keys(map(getpublickey, private_keys))
    return zip(private_keys, public_keys)

def groupkeys(nbits=2048):
    d = dict(zip(['bob', 'alice', 'eve'], generate_keys(3)))
    d_oaep = {x: map(oaep.new, a) for x, a in d.iteritems()}
    d_pss = {x: map(pss.new, a) for x, a in d.iteritems()}
    return d_oaep, d_pss

def slicepubkeys(oaep_keys):
    #extract only the public keys from the dictionary
    puboaep = {x: t[1] for x, t in oaep_keys[0].iteritems()}
    pubpss = {x: t[1] for x, t in oaep_keys[1].iteritems()}
    return puboaep, pubpss

def encrypt(message, oaep_keys, origin, dest):
    oaep, pss = oaep_keys
    
    # bob encrypts and signs the message for alice
    mhash = sha.new(message)
    try:
        #use the PUBLIC key of the destination
        m_encrypted = oaep[dest][1].encrypt(message)
        
        # sign the hash
        # Use the PRIVATE key of the origin
        m_signed = pss[origin][0].sign(mhash)
        return m_encrypted, m_signed
    except ValueError:
        print "Message length is too long for RSA modulus of size ", oaep[dest]._key.size()
        return None
    
def bob2alice(message, oaep_keys):
    return encrypt(message, oaep_keys, 'bob', 'alice')

def decrypt(m_encrypted, oaep_keys, origin, dest):
    # decrypt the message and verify that it came from the origin
    oaep, pss = oaep_keys
    
    #try to decrypt the encrypted message
    try:
        #use the PRIVATE key of dest to decrypt
        m_decrypted = oaep[dest][0].decrypt(m_encrypted[0])
        
        # Hash the decrypted message
        mhash = sha.new(m_decrypted)
        # Compare the hash of the decrypted message with the provided signature
        m_verified = pss[origin][1].verify(mhash, m_encrypted[1])
    except ValueError:
        print "This message probably wasn't intended for you."
        return
    
    return m_decrypted, m_verified

def print_msg(m_encrypted):
    #print the encrypted message as a specially formatted string
    out = ("----BEGIN MESSAGE----",
           m_encrypted[0],
           "----END MESSAGE----",
           "----BEGIN SIGNATURE----",
           m_encrypted[1],
           "----END SIGNATURE----")
    return '\n'.join(out)

def read_msg(m_string):
    d1 = m_string.find('----BEGIN MESSAGE----') + 21
    d2 = m_string.find('----END MESSAGE----', d1)
    
    d3 = m_string.find('----BEGIN SIGNATURE----', d2) + 23
    d4 = m_string.find('----END SIGNATURE----', d3)
    message  = m_string[d1:d2]
    sign = m_string[d3:d4]
    return message.strip(), sign.strip()
