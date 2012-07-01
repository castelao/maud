from numpy import ma

x =  np.range(100)
y = 5*np.sin(2*pi*x/43.) + 3*np.sin(2*pi*(x-7.)/12.) + 2*np.random.rand(len(x))-1


l1=10
l2 = 30 
yf = ma.masked_all(y.shape)
y2f = ma.masked_all(y.shape)

for i in np.arange(len(x)):
    w1 = blackman(x-x[i],l1)
    w1 = w1/sum(w1)
    yf[i] = sum(w1*y) 
    w2 = blackman(x-x[i],l2)
    w2 = w2/sum(w2)
    y2f[i] = sum(w2*y) 


import pylab

pylab.figure()
pylab.plot(x,y)
pylab.plot(x,yf,'r')
pylab.plot(x,y2f,'og')
pylab.plot(x,y-y2f,'k')
pylab.show()
