% Solutions to problem 1

def getFrame(m):
    coeffs = []
    k_max = int(sp.pi*2**(m+1))
    for k in xrange(k_max):
        coeffs.append(-2**m*(sp.cos((k+1)*2**(-m)) - sp.cos(k*2**(-m))))
    coeffs.append(-2**m*(1-sp.cos(k_max*2**(-m))))
    return coeffs

frame_4 = getFrame(4)
frame_6 = getFrame(6)
frame_8 = getFrame(8)

% Here's how to plot each one:
% plt.plot([x*2*sp.pi/len(frame_i) for x in range(len(frame_i))],frame_i,drawstyle='steps')

% Solutions to problem 2

def getDetail(m):
    coeffs = []
    k_max = int(sp.pi*2**(m+1))
    for k in xrange(k_max):
        coeffs.append(-2**m*(2*sp.cos((2*k+1)*2**(-m-1)) - sp.cos((k+1)*2**(-m)) - sp.cos(k*2**(-m))))
        print -2**m*(2*sp.cos((2*k+1)*2**(-m-1)) - sp.cos((k+1)*2**(-m)) - sp.cos(k*2**(-m)))
    if (2*sp.pi < (2*k_max+1)*2**(-m-1)):
        coeffs.append(-2**m*(1-sp.cos(k_max*2**(-m))))
    else:
        coeffs.append(-2**m*(2*sp.cos((2*k_max+1)*2**(-m-1)) - 1 - sp.cos(k*2**(-m))))
    return coeffs

detail = getDetail(4)

b = []
for i in detail:
    b.extend([i,i])
plt.plot([x*2*sp.pi/len(b) for x in range(len(b))],[(-1)**i*b[i] for i in range(len(b))],drawstyle='steps')
