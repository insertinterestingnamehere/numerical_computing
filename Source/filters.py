import scipy as sp

def filter(img, filter=None):
    """Apply filter to img"""

    rows, cols = img.shape
    if rows%2!=0 or cols%2!=0 :
        raise ValueError("filter must have odd number of rows and columns")

    output = sp.zeros((rows*2,cols*2),dtype='float32')
    img = img.astype('float32')
    for i in range(rows):
        for j in range(cols):
            output[i,j] = img[i,j]
    return output
