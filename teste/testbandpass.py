import numpy as np
from numpy import ma, pi
from maud import window_1Dmean, window_1Dbandpass
import pylab

method = 'hann'

x =  ma.array(np.sort(100*np.random.random(100)))
y1 = 3*np.sin(2*pi*(x-7.)/20.)
y2 = 5*np.sin(2*pi*x/60.)
#y3 = 3*np.sin(2*pi*(x+25.)/100.)

Y = y1 + y2 + 1*(np.random.randn(len(x)))


f1 = window_1Dbandpass(data=Y, lshortpass=35, llongpass=8, t=x, method = method, axis=0, parallel=False)
f2 = window_1Dbandpass(data=Y, lshortpass=90, llongpass=40, t=x, method = method, axis=0, parallel=False)
f3 = window_1Dbandpass(data=Y, lshortpass=90, llongpass=8, t=x, method = method, axis=0, parallel=False)


pylab.plot(x,Y, 'o')
pylab.plot(x,y1, 'k--', lw=2)
pylab.plot(x,y2, 'k-.', lw=2)
pylab.plot(x,y1+y2, 'k', lw=2)


pylab.plot(x,f1, 'r', lw=1)
pylab.plot(x,f2, 'g', lw=1)
pylab.plot(x,f3, 'b', lw=1)
pylab.axhline(y=Y.mean(), c='0.7')
pylab.show()
