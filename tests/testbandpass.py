import numpy as np
from numpy import ma, pi
from maud import window_1Dmean, window_1Dbandpass
import pylab

method = 'hann'

N = 1000

x =  ma.array(np.sort(N*np.random.random(N)))
y1 = 3*np.sin(2*pi*(x-7.)/20.)
y2 = 5*np.sin(2*pi*x/60.)
#y3 = 3*np.sin(2*pi*(x+25.)/100.)

Y = y1 + y2 + 1*(np.random.randn(len(x)))


f1 = window_1Dbandpass(data=Y, lshortpass=35, llongpass=8, t=x, method = method, axis=0, parallel=False)
f2 = window_1Dbandpass(data=Y, lshortpass=90, llongpass=40, t=x, method = method, axis=0, parallel=False)
f3 = window_1Dbandpass(data=Y, lshortpass=90, llongpass=8, t=x, method = method, axis=0, parallel=False)


pylab.plot(x[:100],Y[:100], 'o')
pylab.plot(x[:100],y1[:100], 'k--', lw=2)
pylab.plot(x[:100],y2[:100], 'k-.', lw=2)
pylab.plot(x[:100],y1[:100]+y2[:100], 'k', lw=2)


pylab.plot(x[:100],f1[:100], 'r', lw=1)
pylab.plot(x[:100],f2[:100], 'g', lw=1)
pylab.plot(x[:100],f3[:100], 'b', lw=1)
pylab.axhline(y=Y.mean(), c='0.7')


pylab.figure()
T=1./np.fft.fftfreq(N)
ff = np.fft.fft(Y)
ff1 = np.fft.fft(f1)
ff2 = np.fft.fft(f2)
ff3 = np.fft.fft(f3)

pylab.plot(T[1:N/2], ma.absolute(ff[1:N/2]), 'k')
pylab.plot(T[1:N/2], ma.absolute(ff1[1:N/2]),'r')
pylab.plot(T[1:N/2], ma.absolute(ff2[1:N/2]),'g')
pylab.plot(T[1:N/2], ma.absolute(ff3[1:N/2]),'b')
pylab.show()

pylab.show()



