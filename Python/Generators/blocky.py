def blocky3d(A, height, width, depth, offset=(0,0,0), index=False):
    X, Y, Z = A.shape

    if offset > (0,0,0):
        offx, offy, offz = offset
        nx = (X-height)/float(offx) + 1
        ny = (Y-width)/float(offy) + 1
        nz = (Z-depth)/float(offz) + 1
    elif offset == (0,0,0):
        #disjoint blocks
        nx = X/float(height)
        ny = Y/float(width)
        nz = Z/float(depth)
        offx = height
        offy = width
        offz = depth
    else:
        raise ValueError("offset must be non-negative!")

    if int(nx) != nx or int(ny) != ny or int(nz) != nz:
        raise ValueError("Invalid offset!")

    x = y = z = 0
    if not index:
        for m in xrange(int(nx)):
            xheight = x+height
            for n in xrange(int(ny)):
                ywidth = y+width
                for o in xrange(int(nz)):
                    yield A[x:xheight, y:ywidth, z:z+depth]
                    z += offz
                y += offy
                z = 0
            x += offx
            y = 0
    else:
        for m in xrange(int(nx)):
            xheight = x+height
            for n in xrange(int(ny)):
                ywidth = y+width
                for o in xrange(int(nz)):
                    yield A[x:xheight, y:ywidth, z:z+depth], (x, y, z)
                    z += offz
                y += offy
                z = 0
            x += offx
            y = 0


def blocky2d(A, height, width, offset=(0,0), index=False):
    X, Y = A.shape


    if offset > (0,0):
        offx, offy = offset
        nx = (X-height)/float(offx) + 1
        ny = (Y-width)/float(offy) + 1
    elif offset == (0,0):
        #disjoint blocks
        nx = X/float(height)
        ny = Y/float(width)
        offx = height
        offy = width
    else:
        raise ValueError("offset must be non-negative!")

    if int(nx) != nx or int(ny) != ny:
        raise ValueError("Invalid offset!")

    x = y = 0
    if not index:
        for m in xrange(int(nx)):
            xheight = x+height
            for n in xrange(int(ny)):
                yield A[x:xheight, y:y+width]
                y += offy
            x += offx
            y = 0
    else:
        for m in xrange(int(nx)):
            xheight = x+height
            for n in xrange(int(ny)):
                yield A[x:xheight, y:y+width], (x, y)
                y += offy
            x += offx
            y = 0

def blocky1d(A, width, offset=0, index=False):
    X = A.shape[0]

    if offset > 0:
        nx = (X-width)/float(offset) + 1
        offx = offset
    elif offset == 0:
        #disjoint blocks
        nx = X/float(width)
        offx = width
    else:
        raise ValueError("offset must be non-negative!")

    if int(nx) != nx:
        raise ValueError("Invalid offset!")

    x = 0
    if not index:
        for m in xrange(int(nx)):
            yield A[x:x+width]
            x += offx
    else:
        for m in xrange(int(nx)):
            yield A[x:x+width], x
            x += offx
