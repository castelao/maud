
# Gui, 27-06-2012
# Just an idea. Create a class of filtered data. A fake Masked Array object
#   which gives the output on demand, filtered.

try:
    import multiprocessing
    import multiprocessing as mp
except:
    print "I couldn't import multiprocessing"

import numpy as N
import numpy
import numpy as np
from numpy import ma

#try:
#    from maud.cwindow_func import window_func
#except:
#    from window_func import window_func
from window_func import window_func


"""
"""

DEG2RAD = (2*np.pi/360)
RAD2DEG = 1/DEG2RAD
DEG2MIN = 60.
DEG2NM  = 60.
NM2M   = 1852.    # Defined in Pond & Pickard p303.

def find_closer_then(lat, lon, lat_c, lon_c, llimit, method="simplest"):
    """
    """
    ddeg = llimit/(DEG2NM*NM2M)
    possible = np.nonzero((lat<(lat_c+ddeg)) & (lat>(lat_c-ddeg)) & (lon<(lon_c+ddeg)) & (lon>(lon_c-ddeg)))
    L = _distance_1D(lat[possible], lon[possible], lat_c, lon_c)
    ind_closer = L<llimit
    L = L[ind_closer]
    ind = []
    for p in possible:
        ind.append(p[ind_closer])

    return ind, L


def _distance_1D(lat, lon, lat_c, lon_c):
    """
    """
    I = lat.shape[0]
    scale = 0.5*np.pi/180
    deg2m = DEG2NM*NM2M
    L = np.zeros(I)
    fac = np.cos((lat_c+lat)*scale)
    L = ((lat-lat_c)**2+((lon-lon_c)*fac)**2)**.5 \
            * deg2m
    return L
# ============================================================================


def window_1Dmean(data, l, t=None, method='hann', axis=0, parallel=True):
    """ A moving window mean filter, not necessarily a regular grid.

        1D means that the filter is applied to along only one
          of the dimensions, but in the whole array. For example
          in a 3D array, each latXlon point is filtered along the
          time.

        It's not optimized for a regular grid.

        Input:
            - data: np.array or ma.maked_array, nD
            - l: is the size of the filter, in the same unit
                of the t.
	    - t: is the scale of the choosed axis, 1D. If not
                defined, it will be considered a sequence.
            - method: ['hann', 'hamming', 'blackman']
                Defines the weight function type
            - axis: Dimension which the filter will be applied
            - parallel: [True] Will apply the filter with
                multiple processors.
    """
    assert axis <= data.ndim, "Invalid axis!"

    # If necessary, move the axis to be filtered for the first axis
    if axis != 0:
        data_smooth = window_1Dmean(data.swapaxes(0, axis),
            l = l,
            t = t,
            method = method,
            axis = 0,
            parallel = parallel)

        return data_smooth.swapaxes(0, axis)
    # Bellow here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = np.arange(data.shape[axis])

    assert t.shape == (data.shape[axis],), "Invalid size of t."

    # ----
    winfunc = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    if data.ndim==1:
        # It's faster than getmaskarray
        (I,) = np.nonzero(np.isfinite(data))
        for i in I:
            data_smooth[i] = _apply_window_1Dmean(i, t, l, winfunc, data)

    elif parallel is True:
        import multiprocessing as mp
        npes = 2 * mp.cpu_count()
        pool = mp.Pool(npes)
        results = []
        I = data.shape[1]
        for i in range(I):
            results.append(pool.apply_async(window_1Dmean, \
                    (data[:,i], l, t, method, 0, False)))
        pool.close()
        for i, r in enumerate(results):
            data_smooth[:,i] = r.get()
        pool.terminate()

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

def _apply_window_1Dmean(i, t, l, winfunc, data):
    """ Effectively apply 1D moving mean along 1D array
        Support function to window_1Dmean
    """
    dt = t-t[i]
    ind = np.nonzero((np.absolute(dt)<l))
    w = winfunc(dt[ind],l)
    return (data[ind]*w).sum()/(w.sum())


def window_mean_2D_latlon(Lat, Lon, data, l, method='hamming', interp='False'):
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
          - interp: if False, estimate only for the gridponits
              of valid data, i.e. data.mask = output.mask. If
              True, estimate for all gridpoints that has at
              least one valida data point inside the l/2 radius 

        Output:

        !!!ATENTION!!!
        - Might be a good idea to eliminate the dependence on
          fluid.
    """
    assert ((type(l) == float) or (type(l) == int)), \
        "The filter scale (l) must be a float or an int"

    if type(data) == dict:
        output = {}
        for k in data.keys():
            output[k] = window_mean_2D_latlon(Lat, Lon, data[k], l, method)
        return output

    assert data.ndim == 2, "Sorry, for now I'm only handeling 2D arrays"

    weight_func = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    if interp == True:
        I, J = data.shape
        I, J = np.meshgrid(range(I), range(J))
        I = I.reshape(-1); J = J.reshape(-1)
    else:
        I, J = np.nonzero(np.isfinite(data))

    for i, j in zip(I, J):
            ind, r = find_closer_then(Lat, Lon, Lat[i,j], Lon[i,j], llimit=l)
            if len(r) > 0:
                w = weight_func(r, l)
                if data.ndim == 2:
                        #good = np.nonzero(data[ind])
                        tmp = data[ind]*w
                        # There is a problem here. In the case of valid
                        #   but zero value, should be used anyways.
                        wsum = w[np.nonzero(tmp)].sum()
                        if wsum != 0:
                            data_smooth[i,j] = (tmp).sum()/wsum
                #elif data.ndim == 3:
                #    for k in range(data.shape[0]):
                #        data_smooth[k,i,j] = (data[k][ind]*w).sum()/w[good].sum()
                #    else:
                #        data_smooth[k,i,j] = (data[k][ind]*w).sum()/w[good].sum()
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

