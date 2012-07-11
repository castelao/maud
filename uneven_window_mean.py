"""Window means"""

# Gui, 27-06-2012
# Just an idea. Create a class of filtered data. A fake Masked Array object
#   which gives the output on demand, filtered.

import numpy as N
import numpy
import numpy as np
from numpy import ma
from fluid.common.distance import distance
import window_func

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
    if method == 'hamming':
        weight_func = window_func._weight_hamming

    I,J = Lat.shape
    data_smooth={}
    for key in data.keys():
        data_smooth[key] = ma.masked_all(data[key].shape)
    for i in range(I):
        for j in range(J):
            r = distance(Lat,Lon,Lat[i,j],Lon[i,j])
            ind = r<l
            w = weight_func(r[ind],l)
            for key in data.keys():
                if len(data[key].shape)==2:
                    # Stupid solution!!! Think about a better way to do this.
                    if not hasattr(data[key][i,j],'mask'):
                        data_smooth[key][i,j] = (data[key][ind]*w).sum()/w.sum()
                    else:
                        if data[key][i,j].mask==False:
                            data_smooth[key][i,j] = (data[key][ind]*w).sum()/w.sum()
                elif len(data[key].shape)==3:
                    for k in range(data[key].shape[0]):
                        if not hasattr(data[key][k,i,j],'mask'):
                            data_smooth[key][k,i,j] = (data[key][k][ind]*w).sum()/w.sum()
                        else:
                            if data[key].mask[k,i,j]==False:
                                data_smooth[key][k,i,j] = (data[key][k][ind]*w).sum()/w.sum()
    return data_smooth



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

def window_1Dmean(data,l,t=None,method='hann',axis=0):
    """ A moving window mean filter, not necessarily a regular grid.

        1D means that the filter is applied to along only one
          of the dimensions, but in the whole array. For example
          in a 3D array, each latXlon point is filtered along the
          time.

        It's not optimized for a regular grid.

	t is the scale of the choosed axis

        l is the size of the filter.
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
    data_smooth = ma.masked_all(data.shape)

    if method == 'hann':
        winfunc = window_func._weight_hann
    elif method == 'blackman':
        winfunc = window_func._weight_blackman
    
    if len(data.shape)==1:
        #(I,) = np.nonzero(np.isfinite(data))
        (I,) = np.nonzero(~data.mask)
	for i in I:
                dt = t-t[i]
                ind = numpy.absolute(dt)<l
                w = winfunc(dt[ind],l)
                data_smooth[i] = (data[ind]*w).sum()/(w[data[ind].mask==False].sum())
    elif len(data.shape)==2:
        (I,J) = data.shape
	for i in range(I):
	    for j in range(J):
                if data.mask[i,j]==False:
		    if axis == 0:
                        dt = t-t[i]
                        ind = numpy.absolute(dt)<l
                        w = winfunc(dt[ind],l)
                        data_smooth[i,j] = (data[ind,j]*w).sum()/(w[data[ind,j].mask==False].sum())
		    elif axis == 1:
                        dt = t-t[j]
                        ind = numpy.absolute(dt)<l
                        w = winfunc(dt[ind],l)
                        data_smooth[i,j] = (data[ind,j]*w).sum()/(w[data[ind,j].mask==False].sum())
    elif len(data.shape)==3:
        (I,J,K) = data.shape
	for i in range(I):
	    if (data.mask[i]==False).any():
                for j in range(J):
	            if (data.mask[i,j]==False).any():
                        #for k in range(K):
                        for k in numpy.arange(K)[data.mask[i,j]==False]:
                            #if data.mask[i,j,k]==False:
                            if axis == 0:
                                dt = t-t[i]
                                ind = numpy.absolute(dt)<l
                                w = _weight_hann(dt[ind],l)
                                data_smooth[i,j,k] = (data[ind,j,k]*w).sum()/(w[data[ind,j,k].mask==False].sum())
                            elif axis == 1:
                                dt = t-t[j]
                                ind = numpy.absolute(dt)<l
                                w = _weight_hann(dt[ind],l)
                                data_smooth[i,j,k] = (data[ind,j,k]*w).sum()/(w[data[ind,j,k].mask==False].sum())
                            elif axis == 2:
                                dt = t-t[k]
                                ind = numpy.absolute(dt)<l
                                w = _weight_hann(dt[ind],l)
                                data_smooth[i,j,k] = (data[ind,j,k]*w).sum()/(w[data[ind,j,k].mask==False].sum())

    return data_smooth


def window_mean_1D_grid(data,l,method='hann',axis=0):
    print "ATENTION, update the code to call window_1Dmean_grid instead"
    return window_1Dmean_grid(data,l,method=method,axis=axis)

def window_1Dmean_grid(data,l,method='hann',axis=0):
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
    if method == 'hanning':
        method = 'hann'

    r = numpy.arange(numpy.floor(l)+1)
    r = numpy.append(-numpy.flipud(r[1:]),r)
    if method == 'hann':
        #r = numpy.arange(numpy.floor(ll)+1)
	#r = numpy.append(-numpy.flipud(r[1:]),r)
        w = _weight_hann(r,l)
        #w=_weight_hann(numpy.append(-numpy.flipud(numpy.arange(l)[1:]),numpy.arange(l)),l)
    elif method == 'hamming':
        w = _weight_hamming(r,l)
    if axis==0:
        if len(w)>data.shape[0]:
	    print "The filter is longer than the time series. Sorry, I'm not ready to handle that."
	    return
    #data_smooth={}
    #for key in data:
    #    data_smooth[key] = ma.masked_all(data[key].shape)
    data_smooth = ma.masked_all(data.shape)
    #for key in data:
    #if len(data.shape)==2:
    #    I,J = data.shape
    #    if axis==0:
    #        for j in range(J):
    #          data_smooth[:,j] = numpy.convolve(ssh[:,j],w/w.sum(),mode='same')
    #elif len(data.shape)==3:
    if len(data.shape)==1:
        if axis!=0:
	    print "Wait a minute. This is an 1D array, axis need to be equal to 0"
	    return
	I = data.shape
	norm=numpy.convolve(numpy.ones(I),w,mode='same')
	data_smooth[:] = numpy.convolve(data[:],w,mode='same')/norm
    elif len(data.shape)==2:
        I,J = data.shape
        if axis==0:
            norm=numpy.convolve(numpy.ones(I),w,mode='same')
            for j in range(J):
                data_smooth[:,j] = numpy.convolve(data[:,j],w,mode='same')/norm
	elif axis==1:
            norm=numpy.convolve(numpy.ones(J),w,mode='same')
            for i in range(I):
                data_smooth[i,:] = numpy.convolve(data[i,:],w,mode='same')/norm
    elif len(data.shape)==3:
        I,J,K = data.shape
        if axis==0:
            norm=numpy.convolve(numpy.ones(I),w,mode='same')
            for j in range(J):
                for k in range(K):
                    data_smooth[:,j,k] = numpy.convolve(data[:,j,k],w,mode='same')/norm
            #for i in range(l):
            #    data_smooth[i,j,k]=data_smooth[i,j,k]*w.sum()/w[(l-i):].sum()
        #if type(data)==numpy.ma.core.MaskedArray:
        data_smooth.mask = data.mask
        #    data_smooth[data_smooth.data>1e19]=1e20
        #    data_smooth[data_smooth.data>1e19].mask=True
    else:
        print "Sorry, incomplete function. Works only for input arrays with 3D or less."
        return
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


