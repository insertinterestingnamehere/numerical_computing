# Solutions for the Wavelet Lab contained in Haar.tex.
import scipy as sp
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
import pywt
import numpy as np
from scipy.misc import imread
import matplotlib.cm as cm
import Queue
import bitstring as bs
import scipy.misc
from itertools import izip

def dwt(X, L, H, n):
    '''
    Compute the discrete wavelet transform of f with respect to 
    the wavelet filters lo and hi.
    Inputs:
        X -- numpy array corresponding to the signal
        L -- numpy array giving the lo-pass filter
        H -- numpy array giving the hi-pass filter
        n -- integer, giving what level of decomposition
    Returns:
        list of the form [A, D1, D2, ..., Dn] where each entry
        is a numpy array. These are the approximation frame (A)
        and the detail coefficients.
    '''
    coeffs = []
    A = X
    i=0
    while i < n:
        D = fftconvolve(A,H)[1::2]
        A = fftconvolve(A,L)[1::2]
        coeffs.append(D)
        i += 1
    coeffs.append(A)
    coeffs.reverse()
    return coeffs

def idwt(t, lo, hi):
    '''
    Compute the inverse discrete wavelet transform of a list of 
    transform coefficients with respect to the wavelet filters
    lo and hi.
    Inputs:
        t -- a list containing the frame and detail coefficients of
             a signal, corresponding to the output of dwt.
        lo -- numpy array giving the lo-pass filter
        hi -- numpy array giving the hi-pass filter     
    Outputs:
        f -- a numpy array giving the recovered signal.
    '''
    f = t[0]
    for i in xrange(len(t)-1):
        det = t[i+1]
        frame = np.zeros(len(f)*2)
        frame[::2] = f
        frame = fftconvolve(frame, lo)[:-1]
        detail = np.zeros(len(det)*2)
        detail[::2] = det
        detail = fftconvolve(detail, hi)[:-1]
        f = detail + frame
    return f

class huffmanLeaf():
    def __init__(self, symbol):
        self.symbol = symbol
    def makeMap(self, huff_map, path):
        huff_map[self.symbol] = path

class huffmanNode():
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def makeMap(self, huff_map, path):
        """
        Traverse the huffman tree to build the encoding map.
        """
        self.left.makeMap(huff_map, path + '0')
        self.right.makeMap(huff_map, path + '1')
    
def huffman(freqs):
    """
    Generate the huffman tree for the given weights. Return the map.
    """
    q = Queue.PriorityQueue()
    for i in xrange(len(freqs)):
        leaf = huffmanLeaf(i)
        q.put((freqs[i], leaf))
    while q.qsize() > 1:
        l1 = q.get()
        l2 = q.get()
        weight = l1[0] + l2[0]
        node = huffmanNode(l1[1], l2[1])
        q.put((weight,node))
    root = q.get()[1]
    huff_map = dict()
    root.makeMap(huff_map, '')
    return huff_map
    
class WSQ:
    """
    Perform compression using the Wavelet Scalar Quantization algorithm.
    All class attributes are set to None in __init__, but their values
    are initialized in the compress method.
    
    Attributes
    ----------
    _pixels : int, number of pixels in source image
    _s : float, scale parameter for image preprocessing
    _m : float, shift parameter for image preprocessing
    _Q : numpy array, quantization parameters q for each subband
    _Z : numpy array, quantization parameters z for each subband
    _bitstrings : list of 3 BitArrays, giving bit encodings for each group.
    _tvals : tuple of 3 lists of bools, indicating which subbands in each 
             groups were encoded
    _shapes : tuple of 3 lists of tuples, giving shapes of each subband in each group
    _huff_maps : list of 3 dicts, mapping huffman index to bit pattern
    """
    def __init__(self):
        self._pixels = None
        self._s = None
        self._m = None
        self._Q = None
        self._Z = None
        self._bitstrings = None
        self._tvals = None
        self._shapes= None
        self._huff_maps = None
        
    def compress(self, img, r, gamma=2.5):
        """
        The main compression routine. It computes and stores bitstring representation
        of compressed image, along with other values needed for decompression.
        
        Parameters
        ----------
        img : numpy array containing 8-bit integer pixel values
        """
        # pre-processing
        self._pixels = img.shape[0]*img.shape[1]
        self._m = img.mean()
        self._s = max(img.max()-self._m, self._m-img.min())/float(128)
        img = self._preProcess(img)
        
        # subband decomposition
        subbands = self._decompose(img)
        
        # quantization
        self._Q, self._Z = self._getBins(subbands, r, gamma)
        q_subbands = [self._quantize(subbands[i],self._Q[i],self._Z[i]) for i in xrange(len(subbands))]
        # grouping
        groups, self._shapes, self._tvals = self._group(q_subbands)
        
        # for each group, get huffman indices, create huffman tree, and encode
        huff_maps = []
        bitstrings = []
        for i in xrange(3):
            inds, freqs, extra = self._huffmanIndices(groups[i])
            huff_map = huffman(freqs)
            huff_maps.append(huff_map)
            bitstrings.append(self._encode(inds, extra, huff_map))

        # store the bitstrings and the enoding maps
        self._bitstrings = bitstrings
        self._huff_maps = huff_maps 
        
    def decompress(self):
        """
        Return the uncompressed image recovered from the compressed bistring representation.

        Returns
        -------
        img : numpy array giving the recovered uncompressed image.
        """
        # decode the bits, map from indices to coefficients
        groups = []
        for i in xrange(3):
            indices, extras = self._decode(self._bitstrings[i], self._huff_maps[i])
            groups.append(self._indicesToCoeffs(indices, extras))

        # recover the subbands from the groups of coefficients
        q_subbands = self._ungroup(groups, self._shapes, self._tvals)

        # dequantize the subbands
        subbands = [self._dequantize(q_subbands[i], self._Q[i], self._Z[i]) for i in xrange(64)]

        # recreate the image
        img = self._recreate(subbands)

        # post-process
        return self._postProcess(img)
    
    def getRatio(self):
        """
        Calculate the compression ratio achieved.

        Returns
        -------
        ratio : float, the ratio of number of bytes in original image to number of bytes contained
                in the three bitstrings combined.
        """
        return 8*self._pixels/float(self._bitstrings[0].len + self._bitstrings[1].len + self._bitstrings[2].len)
        
    def _preProcess(self, img):
        return (img-self._m)/self._s
    
    def _postProcess(self, img):
        return self._s*img + self._m
    
    def _decompose16(self, image, wavelet):
        """
        Decompose an array into 16 subbands.
        
        Parameters
        ----------
        image : numpy array to be decomposed.
        wavelet : string, giving the pywavelets name of the wavelet to use
        
        Returns
        -------
        subbands : list of 16 numpy arrays giving the subbands
        """
        subbands = []
        LL, HVD = pywt.dwt2(image, wavelet, mode='per')
        dec = pywt.dwt2(LL, wavelet, mode='per')
        subbands.append(dec[0])
        subbands.extend(dec[1])
        for i in xrange(3):
            dec = pywt.dwt2(HVD[i], wavelet, mode='per')
            subbands.append(dec[0])
            subbands.extend(dec[1])
        return subbands
    def _recreate16(self, subbands, wavelet):
        """
        Recreate the original from the 16 subbands.
        
        Parameters
        ----------
        subbands : list of 16 numpy arrays giving the subbands
        wavelet : string, giving the pywavelets name of the wavelet to use
        
        Returns
        -------
        img : numpy array, inverting the effect of _decompose16 
        """
        LL = pywt.idwt2((subbands[0], tuple(subbands[1:4])), wavelet, mode='per')
        details = []
        for i in xrange(1,4):
            details.append(pywt.idwt2((subbands[4*i], tuple(subbands[4*i+1:4*i+4])), wavelet, mode='per'))
        return pywt.idwt2((LL, tuple(details)), wavelet, mode='per')

    def _decompose(self, img):
        """
        Decompose an image into the WSQ subband pattern.
        
        Parameters
        ----------
        img : numpy array holding the image to be decomposed
        
        Returns
        -------
        subbands : list of 64 numpy arrays containing the WSQ subbands in order 
        """
        wavelet='coif1'
        subbands = []
        # first decompose image into 16 subbands
        temp1 = self._decompose16(img, wavelet)

        # next, decompose top left three subbands again into 16
        temp2 = []
        for i in xrange(3):
            temp2.append(self._decompose16(temp1[i], wavelet))

        # finally, decompose top left subband again into 4
        ll, hvd = pywt.dwt2(temp2[0][0], wavelet, mode='per')

        # insert subbands into list in correct order
        subbands.append(ll)
        subbands.extend(hvd)
        subbands.extend(temp2[0][1:])
        subbands.extend(temp2[1])
        subbands.extend(temp2[2])
        subbands.extend(temp1[3:])
        return subbands
  
    def _recreate(self, subbands):
        """
        Recreate an image from the 64 WSQ subbands.
        
        Parameters
        ----------
        subbands : list of 64 numpy arrays containing the WSQ subbands in order
        
        Returns
        -------
        img : numpy array, the recreated image
        """
        wavelet='coif1'
        ll = pywt.idwt2((subbands[0], tuple(subbands[1:4])), wavelet, mode='per')
        temp1 = []
        temp2 = []
        temp2.append(ll)
        temp2.extend(subbands[4:19])
        temp1.append(self._recreate16(temp2, wavelet))
        temp1.append(self._recreate16(subbands[19:35], wavelet))
        temp1.append(self._recreate16(subbands[35:51], wavelet))
        temp1.extend(subbands[51:])
        img = self._recreate16(temp1, wavelet)
        return img
    
    def _getBins(self, subbands, r, gamma):
        """Calculate quantization bin widths for each subband."""
        subband_vars = np.zeros(64)
        fracs = np.zeros(64)
        for i in xrange(len(subbands)): # compute subband variances
            X,Y = subbands[i].shape
            fracs[i]=(X*Y)/(np.float(finger.shape[0]*finger.shape[1]))
            x = np.floor(X/8.)
            y = np.floor(9*Y/32.)
            Xp = np.floor(3*X/4.)
            Yp = np.floor(7*Y/16.)
            mu = subbands[i].mean()
            sigsq = (Xp*Yp-1.)**(-1)*((subbands[i][x:x+Xp, y:y+Yp]-mu)**2).sum()
            subband_vars[i] = sigsq
        A = np.ones(64)
        A[52], A[56] = [1.32]*2
        A[53], A[58], A[55], A[59] = [1.08]*4
        A[54], A[57] = [1.42]*2

        Qprime = np.zeros(64)
        mask = subband_vars >= 1.01
        Qprime[mask] = 10./(A[mask]*np.log(subband_vars[mask]))
        Qprime[:4] = 1
        Qprime[60:] = 0

        K = []
        for i in xrange(60):
            if subband_vars[i] >= 1.01:
                K.append(i)
        while True:
            S = fracs[K].sum()
            P = ((np.sqrt(subband_vars[K])/Qprime[K])**fracs[K]).prod()    
            q = (gamma**(-1))*(2**(r/S-1))*(P**(-1./S))
            E = []
            for i in K:
                if Qprime[i]/q >= 2*gamma*np.sqrt(subband_vars[i]):
                    E.append(i)
            if len(E) > 0:
                for i in E:
                    K.remove(i)
                continue
            break
        Q = np.zeros(64) # final bin widths
        for i in K:
            Q[i] = Qprime[i]/q
        Z = 1.2*Q
        return Q, Z

    def _quantize(self, coeffs, Q, Z):
        """
        Implement a uniform quantizer.
        
        Parameters
        ----------
        coeffs : numpy array containing the floating-point values to be quantized.
        Q : the step size of the quantization, a nonnegative float
        Z : the null-zone width (of the center/0 quantization bin) nonnegative float
        
        Returns
        -------
        out : numpy array of same shape as coeffs holding the quantized values
        """
        out = np.zeros(coeffs.shape)
        if Q == 0:
            return out
        mask1 = coeffs > Z/2.
        mask2 = coeffs < -Z/2.
        out[mask1] = np.floor((coeffs[mask1] - Z/2.)/Q) + 1
        out[mask2] = np.ceil((coeffs[mask2] + Z/2.)/Q) - 1
        return out

    def _dequantize(self, coeffs, Q, Z, C=0.44):
        """
        Reverse the quantization effect (approximately).

        Parameters
        ----------
        coeffs : numpy array of quantized coefficients
        Q : see doc for quantize
        Z : see doc for quantize
        C : centering parameter

        Returns
        -------
        out : array of dequantized coefficients, same shape as coeffs
        """
        out = np.zeros(coeffs.shape)
        if Q == 0:
            return out
        mask1 = coeffs > 0
        mask2 = coeffs < 0
        out[mask1] = (coeffs[mask1] - C)*Q + Z/2.
        out[mask2] = (coeffs[mask2] + C)*Q - Z/2.
        return out

    def _group(self, subbands):
        """
        Split the quantized subbands into 3 groups.

        Parameters
        ----------
        subbands : list of 64 numpy arrays containing quantized coefficients

        Returns
        -------
        gs : tuple (g1,g2,g3)
             each gi a list of quantized coeffs for groups i
        ss : tuple (s1,s2,s3) 
             each si a list of tuples, the shapes of the subbands in group i
        ts : tuple (t1,t2,t3)
             each ti a list of bools indicating which subbands included
        """
        gs = ([],[],[])
        ss = ([],[],[])
        ts = ([],[],[])
        ranges = (xrange(19), xrange(19,52), xrange(52, len(subbands)))
        
        for j in xrange(3):
            for i in ranges[j]:
                ss[j].append(subbands[i].shape)
                if subbands[i].any():
                    gs[j].extend(subbands[i].ravel())
                    ts[j].append(True)
                else:
                    ts[j].append(False)
        
        return gs, ss, ts

    def _ungroup(self, gs, ss, ts):
        """
        Re-create the subband list structure from the three groups.

        Parameters
        ----------
        gs : tuple of form (g1, g2, g3)
        ss : tuple of form (s1, s2, s3)
        ts : tuple of form (t1, t2, t3)
        See the docstring for _group.

        Returns
        -------
        subbands : list of 64 numpy arrays
        """
        subbands = []
        for j in xrange(3): # iterate through the three groups
            i = 0
            for t,shape in izip(ts[j],ss[j]):
                if t: # the subband was transmitted, so grap the coefficients
                    l = shape[0]*shape[1]
                    subbands.append(np.array(gs[j][i:i+l]).reshape(shape))
                    i += l
                else: # the subband wasn't transmitted, so was all zeros
                    subbands.append(np.zeros(shape))
        return subbands
    
    def _huffmanIndices(self, coeffs):
        """
        Calculate the Huffman indices from the quantized coefficients.

        Parameters
        ----------
        coeffs : list of integer values

        Returns
        -------
        inds : list of Huffman indices
        freqs : numpy array whose i-th entry gives frequency of index i
        extra : list of zero run lengths or coefficient magnitudes for exceptional cases
        """
        N = len(coeffs)
        i = 0
        inds = []
        extra = []
        freqs = np.zeros(254)

        # sweep through the quantized coefficients
        while i < N:
            # first handle zero runs
            zero_count = 0
            while coeffs[i] == 0:
                zero_count += 1
                i += 1
                if i >= N:
                    break
            if zero_count > 0 and zero_count < 101:
                inds.append(zero_count - 1)
                freqs[zero_count - 1] += 1
            elif zero_count >= 101 and zero_count < 256: # 8 bit zero run
                inds.append(104)
                freqs[104] += 1
                extra.append(zero_count)
            elif zero_count >= 256: # 16 bit zero run
                inds.append(105)
                freqs[105] += 1
                extra.append(zero_count)
            if i >= N:
                break
            # now handle nonzero coefficients
            if coeffs[i] > 74 and coeffs[i] < 256: # 8 bit pos coeff
                inds.append(100)
                freqs[100] += 1
                extra.append(coeffs[i])
            elif coeffs[i] >= 256: # 16 bit pos coeff
                inds.append(102)
                freqs[102] += 1
                extra.append(coeffs[i])
            elif coeffs[i] < -73 and coeffs[i] > -256: # 8 bit neg coeff
                inds.append(101)
                freqs[101] += 1
                extra.append(abs(coeffs[i]))
            elif coeffs[i] <= -256: # 16 bit neg coeff
                inds.append(103)
                freqs[103] += 1
                extra.append(abs(coeffs[i]))
            else: # current value is a nonzero coefficient in the range [-73, 74]
                inds.append(179 + coeffs[i])
                freqs[179 + coeffs[i]] += 1
            i += 1
        return inds, freqs, extra
    
    def _indicesToCoeffs(self, indices, extra):
        """
        Calculate the coefficients from the Huffman indices plus extra values.

        Parameters
        ----------
        indices : list of integer values (Huffman indices)
        extra : list of indices corresponding to values with exceptional indices

        Returns
        -------
        coeffs : list of quantized coefficients recovered from the indices.
        """
        coeffs = []
        j = 0 # index for extra array
        for s in indices:
            if s < 100: # zero count of 100 or less
                coeffs.extend(np.zeros(s+1))
            elif s == 104 or s == 105: # zero count of 8 or 16 bits
                coeffs.extend(np.zeros(extra[j]))
                j += 1
            elif s in [100, 102]: # 8 or 16 bit pos coefficient
                coeffs.append(extra[j]) # get the coefficient from the extra list
                j += 1
            elif s in [101, 103]: # 8 or 16 bit neg coefficient
                coeffs.append(-extra[j]) # get the coefficient from the extra list
                j += 1
            else: # coefficient from -73 to +74
                coeffs.append(s-179)
        return coeffs
    
    def _encode(self, indices, extra, huff_map):
        """
        Encode the indices using the huffman map, return the resulting bitstring.

        Parameters
        ----------
        indices : list of integer values, the Huffman Indices
        extra : list of integer coefficients corresponding to exceptional indices
        huff_map : dict that maps huffman index to bit pattern

        Returns
        -------
        bits : BitArray object containing bit representation of the huffman indices
        """
        bits = bs.BitArray()
        j = 0 # index for extra array
        for s in indices:
            bits.append('0b' + huff_map[s])
            if s in [104, 100, 101]:
                bits.append('uint:8={}'.format(int(extra[j])))
                j += 1
            elif s in [102, 103, 105]: 
                bits.append('uint:16={}'.format(int(extra[j])))
                j += 1
        return bits
    
    def _decode(self, bits, huff_map):
        """
        Decode the bits using the given huffman map, return the resulting indices.

        Parameters
        ----------
        bits : BitArray object containing the bit-encoded indices
        huff_map : dict that maps huffman index to bit pattern

        Returns
        -------
        indices : list of decoded huffman indices
        extra : list of decoded values corresponding to exceptional indices
        """
        indices = []
        extra = []

        # reverse the huffman map to get the decoding map
        dec_map = {v:k for k, v in huff_map.items()}

        # wrap the bits in an object better suited to reading
        bits = bs.ConstBitStream(bits)

        # read each bit at a time, decoding as we go
        i = 0 # the index of current bit
        pattern = '' # the current bit pattern
        while i < bits.length:
            pattern += bits.read('bin:1') # read in another bit
            i += 1

            # check if current pattern is in the decoding map
            if dec_map.has_key(pattern): 
                indices.append(dec_map[pattern]) # insert huffman index

                # if an exceptional index, read next bits for extra value
                if dec_map[pattern] in (100, 101, 104): # 8-bit int or 8-bit zero run length
                    extra.append(bits.read('uint:8'))
                    i += 8
                elif dec_map[pattern] in (102, 103, 105): # 16-bit int or 16-bit zero run length
                    extra.append(bits.read('uint:16'))
                    i += 16
                pattern = '' # reset the bit pattern
        return indices, extra 
