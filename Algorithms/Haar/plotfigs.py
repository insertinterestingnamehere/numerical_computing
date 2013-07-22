import matplotlib
#matplotlib.rcParams = matplotlib.rc_params_from_file('../../matplotlibrc')

from matplotlib import pyplot as plt
import solution
import scipy as sp

# Generate the sine curve
def sineCurve():
    pts = sp.linspace(0,2*sp.pi,256)
    plt.plot(pts,sp.sin(pts))
    plt.savefig('sinecurve.pdf')
    plt.clf()
    
# Generate the discretized sine curve
def discSineCurve():
    frame_4 = solution.getFrame(2)
    plt.plot(sp.linspace(0,2*sp.pi,len(frame_4)),frame_4,drawstyle='steps')
    plt.savefig('discreteSineCurve.pdf')
    plt.clf()

# Generate the detail for the sine curve
def sineCurveDetail():    
    detail = solution.getDetail(2)
    detail[-1] = detail[-2]
    b = []
    for i in detail:
        b.extend([i,-i])
    plt.plot(sp.linspace(0,2*sp.pi,len(b)),b,drawstyle='steps')
    plt.savefig('sineCurveDetail.pdf')
    plt.clf()

# Generate the Mexican Hat Wavelet image
def mexicanHat():
    def mex(sigma,t):
        return (2.0/sp.sqrt(3*sigma*sp.sqrt(sp.pi)))*(1-(1.0*t**2/sigma**2))*sp.exp(-t**2*1.0/(2*sigma**2))
    x = sp.linspace(-10,10,500)
    plt.plot(x,mex(2,x))
    plt.savefig('mexicanHat.pdf')
    plt.clf()

sineCurve()
discSineCurve()
sineCurveDetail()
mexicanHat()   
    
