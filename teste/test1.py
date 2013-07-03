import numpy as np
from numpy import ma, pi
from maud import window_1Dmean, window_1Dbandpass
import pylab

x =  ma.array(np.sort(100*np.random.random(100)))
y1 = 5*np.sin(2*pi*x/50.)

Y = y1 + 1*(np.random.randn(len(x)))

f1 = window_1Dmean(data=Y, l=0.5, t=x, method='hann', axis=0, parallel=False)
f2 = window_1Dmean(data=Y, l=2, t=x, method='hann', axis=0, parallel=False)
f3 = window_1Dmean(data=Y, l=10, t=x, method='hann', axis=0, parallel=False)


pylab.plot(x,Y, 'o')
pylab.plot(x,y1, 'K', lw=2)
pylab.plot(x,f1, 'r', lw=1)
pylab.plot(x,f2, 'g', lw=1)
pylab.plot(x,f3, 'b', lw=1)
pylab.show()
