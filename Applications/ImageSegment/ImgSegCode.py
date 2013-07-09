#Applications: Image Segmentation

def imgAdj(img,radius, sigmaI, sigmaX):
    nodes = img.flatten()
    height,width=img.shape
    W = spar.lil_matrix((nodes.size, nodes.size), dtype=float)
    
    #Here we generate the values that go into the adjacency matrix W.  For the most part we don't have to worry to 
    #much and we will be in the final else statement.  However, in the case that we are on the boundaries we need to make sure
    #that we aren't looking at pixels that aren't there.  There are four more elif statements needed, fill them in.
            
    for row in xrange(height):
        for col in xrange(width):
            #top right
            rowcol = row * width + col
            
            if (row < radius) and (col < radius):
                for k in xrange(row + radius):
                    for l in xrange(col + radius):
            
            #top left
            elif (row < radius) and (col > width - radius):
                #subMat = img(1:i+r,j-r:width)
                for k in xrange(row + radius):
                    for l in xrange(col - radius, width):
                                    
            #bottom right	
            elif (row > height - radius) and (col < radius):
                #subMat = img(i-r:height,1:j+r);
                for k in xrange(row - radius, height):
                    for l in xrange(col + radius):

            #bottom left
            elif (row > height - radius) and (col > width - radius):
                #subMat = img(i-r:height,j-r:width);
                for k in xrange(row - radius, height):
                    for l in xrange(col - radius, width):

            #top middle
            elif (row < radius):# and (col > radius and col < width-radius):
                for k in xrange(row + radius):
                    for l in xrange(col - radius, col + radius):
  
            #middle left
            elif (col < radius):
                for k in xrange(row - radius, row + radius):
                    for l in xrange(col + radius):

            #middle right
            elif (col > height - radius):
                for k in xrange(row - radius, row + radius):
                    for l in xrange(col - radius, width):
                    
            #bottom middle
            elif (row > height - radius):
                for k in xrange(row - radius, height):
                    for l in xrange(col + radius):
            
            else: # (row > radius and row < height-radius) and (col > radius and col < width-radius):
                #subMat = img(i-r:i+r,j-r:j+r);
                for k in xrange(row - radius, row + radius):
                    for l in xrange(col - radius, col + radius):

    W = W.tocsc()    
    return W
    
