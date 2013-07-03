

# Gui, 27-06-2012
# Just an idea. Create a class of filtered data. A fake Masked Array object
#   which gives the output on demand, filtered.

import multiprocessing

import numpy as N
import numpy
import numpy as np
from numpy import ma

try:
    from fluid.cdistance import distance
    from fluid.cdistance import find_closer_then
except:
    from fluid.common.distance import distance
    from fluid.common.distance import find_closer_then

#try:
#    from maud.cwindow_func import window_func
#except:
#    from window_func import window_func
from window_func import window_func


"""
"""


def window_mean_2D_latlon(Lat, Lon, data, l, method='hamming'):
    """
        Right now I'm doing for Masked Arrays only.
        data should be a dictionary with 2D arrays

        Input:
          - Lat: 2D array with latitudes
          - Lon: 2D array with longitudes
          - data: There are two possibilities, it can be an 
            array of at least 2D, or a dictionary where each
            element is an array of at least 2D.
          - l: window filter size, in meters
          - method: weight function type

        Output:

        !!!ATENTION!!!
        - Might be a good idea to eliminate the dependence on
          fluid.
    """
    weight_func = window_func(method)

    I,J = Lat.shape
    data_smooth={}
    for key in data.keys():
        data_smooth[key] = ma.masked_all(data[key].shape)
    for i in range(I):
        for j in range(J):
            ind, r = find_closer_then(Lat, Lon, Lat[i,j], Lon[i,j], llimit=l)
            w = weight_func(r, l)
            for key in data.keys():
                if len(data[key].shape)==2:
                    good = np.nonzero(data[key][ind])
                    #ind = np.nonzero(data[key])
                    # Stupid solution!!! Think about a better way to do this.
                    if not hasattr(data[key][i,j],'mask'):
                        data_smooth[key][i,j] = (data[key][ind]*w).sum()/w[good].sum()
                    else:
                        if data[key][i,j].mask==False:
                            data_smooth[key][i,j] = (data[key][ind]*w).sum()/w[good].sum()
                elif len(data[key].shape)==3:
                    for k in range(data[key].shape[0]):
                        if not hasattr(data[key][k,i,j],'mask'):
                            data_smooth[key][k,i,j] = (data[key][k][ind]*w).sum()/w[good].sum()
                        else:
                            if data[key].mask[k,i,j]==False:
                                data_smooth[key][k,i,j] = (data[key][k][ind]*w).sum()/w[good].sum()
    return data_smooth

# ==== Bellow here, I need to do some serious work on ====

def window_mean_2D(x, y, z, l, method='hamming'):
    """
    """
    if method == 'hamming':
        weight_func = _weight_hamming_2D

    if len(z.shape) < 2:
        print "At least 2D"

    output = ma.masked_all(z.shape)
    if len(z.shape) > 2:
        for i in range(z.shape[0]):
            output[i] = window_mean_2D(x, y, z[i], method)

    elif len(z.shape) == 2:
        I,J = z.shape
        for i in range(I):
            for j in range(J):
	        w = weight_func((x-x[i,j]), (y-y[i,j]), l)
	        output[i,j] = (z*w).sum()/(w.sum())
        return output



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

def _convolve(x, dt, l, winfunc):
    w = winfunc(dt, l)
    y = (x*w).sum()/(w[x.mask==False].sum())
    return y

def window_1Dbandpass(data, lshorterpass, llongerpass, t=None, method='hann', axis=0, parallel=True):
    """

        Input:
            lshorterpass: shorter wavelenghts are preserved
            llongerpass: longer wavelengths are preserved
    """
    if axis > len(data.shape):
        print "The data array don't contain so many dimensions. Choose another axis"
	return

    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = numpy.arange(data.shape[axis])

    elif t.shape != (data.shape[axis],):
        print "The scale variable t don't have the same size of the choosed axis of the data array"
        return 
    # ----
    #data_smooth = ma.masked_all(data.shape)
    data_smooth = window_1Dmean(data,
                        t = t,
                        l = llongerpass,
                        axis = axis,
                        parallel = False)

    data_smooth = data_smooth - window_1Dmean(data_smooth,
                        t = t,
                        l = lshorterpass,
                        axis = axis,
                        parallel=False)

    return data_smooth


def window_1Dmean(data, l, t=None, method='hann', axis=0, parallel=True):
    """ A moving window mean filter, not necessarily a regular grid.

        1D means that the filter is applied to along only one
          of the dimensions, but in the whole array. For example
          in a 3D array, each latXlon point is filtered along the
          time.

        It's not optimized for a regular grid.

	t is the scale of the choosed axis

        l is the size of the filter.
    """
    assert axis <= data.ndim, "Invalid axis!"

    if axis != 0:
        data_smooth = window_1Dmean(data.swapaxes(0,axis),
            l = l,
            t = t,
            method = method,
            axis=0,
            parallel = parallel)

        return data_smooth.swapaxes(0,axis)

    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = numpy.arange(data.shape[axis])

    elif t.shape != (data.shape[axis],):
        print "Invalid size of t."
        return 
    # ----
    winfunc = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    if data.ndim==1:
        # It's faster than getmaskarray
        (I,) = np.nonzero(np.isfinite(data))
	for i in I:
            dt = t-t[i]
            ind = numpy.nonzero((numpy.absolute(dt)<l))
            w = winfunc(dt[ind],l)
            data_smooth[i] = (data[ind]*w).sum()/(w.sum())

    else:
        I = data.shape[1]
        for i in range(I):
            data_smooth[:,i] = window_1Dmean(data[:,i], 
                    l = l,
                    t = t,
                    method = method, 
                    axis = 0, 
                    parallel=parallel)

    return data_smooth

#    elif len(data.shape)==2:
#        (I,J) = data.shape
#        if parallel == True:
#            nprocesses = 2*multiprocessing.cpu_count()
#            #logger.info("I'll work with %s parallel processes" % nprocesses)
#            filters_pool = multiprocessing.Pool(nprocesses)
#            results = []
#            for j in range(J):
#                results.append(filters_pool.apply_async(window_1Dmean, (data[:,j], l, t, method, axis, parallel)))
#            filters_pool.close()
#            for n, r in enumerate(results):
#                data_smooth[:,n] = r.get()
#        else:
#            for j in range(J):
#                data_smooth[:,j] = window_1Dmean(data[:,j], l, t, method, axis)


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
def get_halfpower_period(data, filtered):
    """ Returns the gain per frequency
    """
    nt,ni,nj = data.shape
    gain = ma.masked_all((nt,ni,nj))
    for i in range(ni):
        for j in range(nj):
	    if ~filtered[:,i,j].mask.all():
	        gain[:,i,j] = numpy.absolute(numpy.fft.fft(filtered[:,i,j]-filtered[:,i,j].mean())) / numpy.absolute(numpy.fft.fft(data[:,i,j]-data[:,i,j].mean()))
    gain_median = ma.masked_all(nt)
    gain_25 = ma.masked_all(nt)
    gain_75 = ma.masked_all(nt)
    # Run for each frequency, which are in the same number of timesteps
    from scipy.stats import scoreatpercentile
    for t in range(nt):
        #gain_median[t] = numpy.median(gain[t,:,:].compressed()[numpy.isfinite(gain[t,:,:].compressed())])
        tmp = gain[t,:,:].compressed()[numpy.isfinite(gain[t,:,:].compressed())]
        gain_median[t] = scoreatpercentile(tmp,50)
        gain_25[t] = scoreatpercentile(tmp,25)
        gain_75[t] = scoreatpercentile(tmp,75)

    freq=numpy.fft.fftfreq(nt)/dt.days

    #from scipy.interpolate import UnivariateSpline
    #s = UnivariateSpline(gain_median[numpy.ceil(nt/2.):], -freq[numpy.ceil(nt/2.):], s=1)
    #xs = -freq[numpy.ceil(nt/2.):]
    #ys = s(xs)

    import rpy2.robjects as robjects
    smooth = robjects.r['smooth.spline'](robjects.FloatVector(gain_median[numpy.ceil(nt/2.):]),robjects.FloatVector(-freq[numpy.ceil(nt/2.):]),spar=.4)
    #smooth = robjects.r['smooth.spline'](robjects.FloatVector(-freq[numpy.ceil(nt/2.):]),robjects.FloatVector(gain_median[numpy.ceil(nt/2.):]),spar=.4)
    s_interp = robjects.r['predict'](smooth,x=0.5)
    halfpower_period = 1./s_interp.rx2['y'][0]

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







#ssh_smooth=window_mean_1D_grid(data['ssh'],l=7)
#pylab.plot(data['ssh'][:,j,k])
#pylab.plot(data_smooth[:,j,k],'r')
#pylab.show()
#
#output=ma.masked_all(x.shape)
#for i in range(l,I-1):
#    output[i]=
#
#x=numpy.arange(10)
#w = window_mean._weight_hann(numpy.arange(2*l+1)-l,l)
#w=numpy.array([1,2,1])
#y=numpy.convolve(x,w,mode='same')
#y.shape
#x
#w
#y




from numpy import pi

#def _weight_hamming_2D(x,y,l):
#    """
#
#    Check it!
#    """
#    r=(x**2+y**2)**0.5
#    w=0.54 - 0.46 * numpy.cos(pi*r/l)
#    w[r>l]=0
#    return w



#function w = window(N,wt)
#%
#%  w = window(N,wt)
#%
#%  generate a window function
#%
#%  N = length of desired window
#%  wt = window type desired
#%       'rect' = rectangular        'tria' = triangular (Bartlett)
#%       'hann' = Hanning            'hamm'  = Hamming
#%       'blac' = Blackman
#%
#%  w = row vector containing samples of the desired window
#nn = N-1;
#pn = 2*pi*(0:nn)/nn;
#if wt(1,1:4) == 'rect',
#                        w = ones(1,N);
#elseif wt(1,1:4) == 'tria',
#                        m = nn/2;
#                        w = (0:m)/m;
#                        w = [w w(ceil(m):-1:1)];
#elseif wt(1,1:4) == 'hann',
#                        w = 0.5*(1 - cos(pn));
#elseif wt(1,1:4) == 'hamm',
#                        w = .54 - .46*cos(pn);
#elseif wt(1,1:4) == 'blac',
#                        w = .42 -.5*cos(pn) + .08*cos(2*pn);
#else
#                        disp('Incorrect Window type requested')
#end


