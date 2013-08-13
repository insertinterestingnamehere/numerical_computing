import numpy as np

def filter(img, filter=None):
    """Apply filter to img"""

    fr, fc = filter.shape
    ir, ic = img.shape

    if fr % 2 == 0 or fc % 2 == 0 :
        raise ValueError("filter must have odd number of rows and columns")

    hfc, hfr = fc/2, fr/2
    t1, t2 = ir+hfr*2, ic+hfc*2
    imgc = np.zeros((t1, t2))
    imgc[hfr:t1-hfr, hfc:t2-hfc] = img

    #broadcast edges out
    imgc[0:hfr, hfc:t2-hfc] = img[0]
    imgc[-hfr:, hfc:t2-hfc] = img[-1]
    imgc[:, -hfc+1:] = imgc[:, [-hfc]]
    imgc[:, :hfc] = imgc[:, [hfc]]

    npsum = np.sum
    output = np.zeros_like(imgc)
    for i in xrange(hfr, t1-hfr):
        for j in xrange(hfc, t2-hfc):
            #check for upper left corner
                output[i, j] = npsum(imgc[i-hfr:i+hfr+1, j-hfc:j+hfc+1] * filter)

    return output
