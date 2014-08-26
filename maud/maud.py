
"""
"""
# Gui, 27-06-2012
# Just an idea. Create a class of filtered data. A fake Masked Array object
#   which gives the output on demand, filtered.

try:
    import multiprocessing as mp
except:
    print "I couldn't import multiprocessing"

import numpy as N
import numpy
import numpy as np
from numpy import ma

from window_func import window_func
from distance import haversine

# ==== Bellow here, I need to do some serious work on ====

def window_mean(y,x=None,x_out=None,method="rectangular",boxsize=None):
    """Windowed means along 1-D array

    Input:
        - x [0,1,2,3,...] =>
        - x_out [x] =>
        - method [rectangular]:
            + rectangular => All data in window have same weight
        - boxsize [mean space] =>
    Output:

    Apply windowed means on a 1-D data array. Selecting adequate x_out
    and boxsize could define boxmeans or smooth filters. Method defines
    the weight method.

    An important point of this function is the ability to work with 
    unhomogenious spaced samples. Data ([1,2,3]) colected at [1,2,4] 
    times would be different if was collected at [1,2,3].
    """
    if(x==None):
        x=N.arange(N.size(y))

    if(x_out==None):
        x_out=x

    #y_out = N.zeros(N.size(x_out),N.float)
    y_out = ma.masked_all(x_out.shape)

    if(boxsize==None):
        # !!! Improve it! A better way than *1. ?!
        boxsize =(max(x)-min(x))/(N.size(x_out)*1.)

    half_boxsize = boxsize/2.

    #for x_i in x_out:
    for i in range(N.size(x_out)):
        x_i = x_out[i]


        # Higher window limit
        hi_limit = x_i+half_boxsize
        # Lower window limit
        lo_limit = x_i-half_boxsize
        # index of values inside window
        index = N.less_equal(x,hi_limit)*N.greater_equal(x,lo_limit)

        # !!! INSERT some type of check for minimum number of samples to be considered

        # x values on window around x_i
        x_tmp = N.compress(index,x)-x_i
        # y values on window
        y_tmp = N.compress(index,y)

        # weights in window according to x position
        weight = window_weight(x_tmp,boxsize,method)

        y_out[i] = N.sum(y_tmp*weight)

    return y_out

# To improve, the right to way to implement these filters are to define the halfpower cutoff, instead of an l dimension. Then the l dimension is defined on the function according to the weightning system for the adequate l.

def window_1Dmean_grid(data, l, method='hann', axis=0, parallel=False):
    """ A moving window mean filter applied to a regular grid.

        1D means that the filter is applied to along only one
          of the dimensions, but in the whole array. For example
          in a 3D array, each latXlon point is filtered along the
          time.

        The other types of filter consider the scales of each
          dimension. On this case it's considered a regular
          grid, so the filter can be based on the number of
          elements, and so be much optimized.

        l is in number of cells around the point being evaluated.
    """
    assert axis <= data.ndim, "Invalid axis!"

    if axis != 0:
        data_smooth = window_1Dmean_grid(data.swapaxes(0, axis), 
                l = l, 
                method = method, 
                axis = 0, 
                parallel = parallel)

        return data_smooth.swapaxes(0, axis)

    winfunc = window_func(method)
    r = np.arange(-np.floor(l/2),np.floor(l/2)+1)
    w = winfunc(r, l)

    data_smooth = ma.masked_all(data.shape)

    I = data.shape[0]
    norm = np.convolve(np.ones(I), w ,mode='same')
    if len(data.shape)==1:
	norm=numpy.convolve(numpy.ones(I),w,mode='same')
	data_smooth[:] = numpy.convolve(data[:],w,mode='same')/norm

    elif len(data.shape) == 2:
        I, J = data.shape
        for j in range(J):
            data_smooth[:,j] = np.convolve(data[:,j], w, mode='same')/norm

    elif len(data.shape) >2:
        I, J = data.shape[:2]
        for j in range(J):
                data_smooth[:,j] = window_1Dmean_grid(data[:,j], 
                        l = l, 
                        method = method, 
                        axis=0, 
                        parallel = parallel)

    try:
        data_smooth.mask = data.mask
    except:
        pass

    return data_smooth


# ----
def get_halfpower_period(data, filtered, dt):
    """ Returns the gain per frequency
    """
    nt, ni, nj = data.shape
    gain = ma.masked_all((nt, ni, nj))
    for i in range(ni):
        for j in range(nj):
	    if ~filtered[:,i,j].mask.all():
	        gain[:,i,j] = np.absolute(np.fft.fft(filtered[:,i,j]-filtered[:,i,j].mean())) / np.absolute(np.fft.fft(data[:,i,j]-data[:,i,j].mean()))
    gain_median = ma.masked_all(nt)
    gain_25 = ma.masked_all(nt)
    gain_75 = ma.masked_all(nt)
    # Run for each frequency, which are in the same number of timesteps
    from scipy.stats import scoreatpercentile
    for t in range(nt):
        #gain_median[t] = numpy.median(gain[t,:,:].compressed()[numpy.isfinite(gain[t,:,:].compressed())])
        #tmp = gain[t,:,:].compressed()[numpy.isfinite(gain[t,:,:].compressed())]
        #gain_median[t] = scoreatpercentile(tmp,50)
        gain_median[t] = ma.median(gain[t])
        #gain_25[t] = scoreatpercentile(tmp,25)
        #gain_75[t] = scoreatpercentile(tmp,75)

    freq = np.fft.fftfreq(nt)/dt.days

    halfpower_period = 1./freq[np.absolute(gain_median-0.5).argmin()]


    #from scipy.interpolate import UnivariateSpline
    #s = UnivariateSpline(gain_median[numpy.ceil(nt/2.):], -freq[numpy.ceil(nt/2.):], s=1)
    #xs = -freq[numpy.ceil(nt/2.):]
    #ys = s(xs)

    #import rpy2.robjects as robjects
    #smooth = robjects.r['smooth.spline'](robjects.FloatVector(gain_median[numpy.ceil(nt/2.):]),robjects.FloatVector(-freq[numpy.ceil(nt/2.):]),spar=.4)
    ##smooth = robjects.r['smooth.spline'](robjects.FloatVector(-freq[numpy.ceil(nt/2.):]),robjects.FloatVector(gain_median[numpy.ceil(nt/2.):]),spar=.4)
    #s_interp = robjects.r['predict'](smooth,x=0.5)
    ##halfpower_period = 1./s_interp.rx2['y'][0]
    #halfpower_period = 1./s_interp.rx2(2)[0]

    #smooth = robjects.r['smooth.spline'](robjects.FloatVector(-freq[numpy.ceil(nt/2.):]),robjects.FloatVector(gain_median[numpy.ceil(nt/2.):]),spar=.4)
    #s_interp = robjects.r['predict'](smooth, x = robjects.FloatVector(-freq[numpy.ceil(nt/2.):]))

    #print "Filter half window size: %s" % l
    #print "Half Power Period: %s" % halfpower_period
    #self.halfpower_period = halfpower_period

    return halfpower_period



#pylab.plot(-freq[numpy.ceil(nt/2.):],  gain_median[numpy.ceil(nt/2.):]  )
#pylab.plot(1./halfpower_period, 0.5,'rx')
#pylab.show()

# In the future implement a check to show the average effect in the
#   spectrum. I.e. estimate fft before and after and compare the difference
#I,J,K = ssh.shape
##fft_input = ma.masked_all((I,J,K))
##fft_output = ma.masked_all((I,J,K))
#tmp = ma.masked_all((I,J,K))
#tmp2 = ma.masked_all(I)
#for j in range(J):
#    for k in range(K):
#        if ssh[:,j,k].mask.any()==False:
#            #fft_input[:,j,k] = numpy.fft.fft(ssh[:,j,k])
#            fft_input = numpy.absolute(numpy.fft.fft(ssh[:,j,k]))
#            #fft_output[:,j,k] = numpy.fft.fft(output[:,j,k])
#            fft_output = numpy.absolute(numpy.fft.fft(output[:,j,k]))
#            tmp[:,j,k] = (fft_input-fft_output)/fft_input
#for i in range(I):
#    tmp2[i] = tmp[i].mean()

