# helper function
def getNeighbors(row, col, radius, height, width):
    '''
    Calculate the indices and corresponding distances of pixels within radius
    of the pixel at (row,col), where the pixels are in a (height, width) shaped
    array. The returned indices are with respect to the flattened version of the
    array. This is a helper function for adjacency.
    Inputs:
        row, col -- denotes the row and column number of the pixel we are 
                    centered at.
        radius -- radius of the circular region centered at pixel (row, col)
        height, width -- the height and width of the original image, in pixels
    Returns:
        indices -- a flat array of indices of pixels that are within distance r
                   of the pixel at (row, col)
        distances -- a flat array giving the respective distances from these 
                     pixels to the center pixel.
    '''
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r+1, width))
    y = np.arange(max(row - r, 0), min(row + r+1, height))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(((X-np.float(col))**2+(Y-np.float(row))**2))
    mask = (R<radius)
    return (X[mask] + Y[mask]*width, R[mask])