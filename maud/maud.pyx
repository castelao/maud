# -*- coding: utf-8 -*-
# cython: profile=True
# filename: maud.pyx

"""

    Wasn't this below supposed to be faster than from i in xrange()?
    #for i from 0 < = i < I:
    #    for j from 0 < = j < J:
"""

import numpy as np
from numpy import ma

import multiprocessing as mp

cimport numpy as np
from libc.math cimport abs, cos, sin, asin, sqrt, M_PI

from maud.cwindow_func_scalar import window_func_scalar
from maud.cdistance import haversine_scalar as _haversine_scalar

try:
    from progressbar import ProgressBar
except:
    print "ProgressBar is not available"


np.import_array()

DTYPE = np.float
ctypedef np.float_t DTYPE_t


from maud.window_func import window_func

def wmean_1D_serial(data, l, t=None, method='hann', axis=0, interp=False):
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
    """
    #assert type(data)) in [np.ndarray, ma.MaskedArray]
    assert axis <= data.ndim, "Invalid axis!"

    # If necessary, move the axis to be filtered for the first axis
    if axis != 0:
        print("Will temporary swapaxes")
        data_smooth = wmean_1D_serial(data.swapaxes(0, axis),
            l = l,
            t = t,
            method = method,
            axis = 0,
            interp = interp)

        return data_smooth.swapaxes(0, axis)
    # Below here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
        t = np.arange(data.shape[axis])

    t = t.astype('float64')
    assert t.shape == (data.shape[axis],), "Invalid size of t."

    l = float(l)

    if (data.ndim == 1):
        wfunc = window_func_scalar(method)

        if (type(data) is np.ndarray):
            data_smooth = convolve_1D_array(data, t, l, wfunc)
            return data_smooth

        elif (type(data) is ma.MaskedArray) and (data.ndim == 1):
            d, m = convolve_1D_MA(data.data,
                    ma.getmaskarray(data).astype('int8'),
                    t, l, method, interp)
            return ma.masked_array(d, m)


    if type(data) is np.ndarray:
        data_smooth = np.empty(data.shape)
    else:
        data_smooth = ma.masked_all(data.shape)

    # ----
    if data.ndim > 1:
        for i in range(data.shape[1]):
            data_smooth[:,i] = wmean_1D_serial(data[:,i],
                l = l,
                t = t,
                method = method,
                axis = 0,
                interp = interp)

        return data_smooth
    # ----
    # Here on it is expected data.ndim == 1

    wfunc = window_func(method)

    if interp is True:
        I = range(len(t))
    else:
        (I,) = np.nonzero(~ma.getmaskarray(data))

    for i in I:
        dt = t - t[i]
        w = wfunc(dt, l)
        ind = (w != 0) & (~ma.getmaskarray(data))
        if ind.any():
            tmp = data[ind]*w[ind]
            wsum = w[ind].sum()
            if wsum != 0:
                data_smooth[i] = (tmp).sum()/wsum

    return data_smooth


def wmean_1D(data, l, t=None, method='hann', axis=0, interp=False):
    return wmean_1D_serial(data, l, t, method, axis, interp)


def window_1Dmean(data, double l, t=None, method='hann', axis=0, parallel=True):
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

    # If necessary, move the axis to be filtered for the first axis
    if axis != 0:
        data_smooth = window_1Dmean(data.swapaxes(0,axis),
            l = l,
            t = t,
            method = method,
            axis=0,
            parallel = parallel)

        return data_smooth.swapaxes(0,axis)
    # Bellow here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
        t = np.arange(data.shape[axis], dtype=np.float)
    else:
        assert t.dtype == np.float, "Var t must be of type np.float"

    # t must has the same shape of data along axis
    assert t.shape == (data.shape[axis],)

    # ----
    weight_func = window_func_scalar(method)

    data_smooth = ma.masked_all(data.shape)

    if data.ndim==1:
        # It's faster than getmaskarray
        (I,) = np.nonzero(np.isfinite(data))

        for i in I:
            #data_smooth[i] = apply_window_1Dmean(data, t[i], t, l, I, weight_func)
            out = apply_window_1Dmean(data, t[i], t, l, I, weight_func)
            if out is not None:
                data_smooth[i] = out

    elif parallel is True:
        npes = 2 * mp.cpu_count()
        pool = mp.Pool(npes)
        results = []
        I = data.shape[1]
        try:
            pbar = ProgressBar(maxval=I).start()
        except:
            pass

        for i in range(I):
            results.append( pool.apply_async( window_1Dmean, \
                           (data[:,i], l, t, method, 0, False)))
        pool.close()
        for i, r in enumerate(results):
            try:
                pbar.update(i)
            except:
                pass
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


def convolve_1D_array(np.ndarray[DTYPE_t, ndim=1] data,
        np.ndarray[DTYPE_t, ndim=1] t, double l, wfunc):

    cdef unsigned int I = len(t)
    cdef unsigned int i, ii
    cdef double D, W, dt, w
    # The ideal would be use data.shape instead of I.
    cdef np.ndarray[DTYPE_t, ndim=1] data_smooth = np.empty(I)

    #wfunc = window_func_scalar(method)

    for i in xrange(I):
        D = 0
        W = 0
        for ii in xrange(I):
            dt = t[ii] - t[i]
            if abs(dt) <= l:
                w = wfunc(dt, l)
                if w != 0:
                    D += data[ii]*w
                    W += w

        assert (W != 0), "It's an array, must have at least one valid data"
        data_smooth[i] = D/W

    return data_smooth


def convolve_1D_MA(np.ndarray[DTYPE_t, ndim=1] data,
        np.ndarray[np.int8_t, ndim=1] mask,
        np.ndarray[DTYPE_t, ndim=1] t, double l, method, bint interp):

    cdef unsigned int I = len(t)
    cdef unsigned int i, ii
    cdef double D, W, dt, w
    # The ideal would be use data.shape instead of I.
    cdef np.ndarray[DTYPE_t, ndim=1] data_smooth = np.empty(I)
    cdef np.ndarray[np.int8_t, ndim=1] mask_smooth = np.zeros(I, 'int8')

    wfunc = window_func_scalar(method)

    for i in xrange(I):
        if (interp == 0) and (mask[i] == 1):
            mask_smooth[i] = 1
        else:
            D = 0
            W = 0
            for ii in xrange(I):
                if mask[ii] == 0:
                    dt = t[ii] - t[i]
                    if abs(dt) <= l:
                        w = wfunc(dt, l)
                        if w != 0:
                            D += data[ii]*w
                            W += w

            assert (W != 0), "It's an array, must have at least one valid data"
            data_smooth[i] = D/W

    return data_smooth, mask_smooth


def apply_window_1Dmean(np.ndarray[DTYPE_t, ndim=1] data,
        double t0, np.ndarray[DTYPE_t, ndim=1] t, double l,
        np.ndarray[np.int_t, ndim=1] I, weight_func):

    cdef int i
    cdef double D, W
    cdef double dt, w
    #cdef size_t I

    W = 0
    D = 0

    for i in I:
        dt = t[i] - t0
        if abs(dt) <= l:
            w = weight_func(dt, l)
            if w != 0:
                D += data[i] * w
                W += w
    if W != 0:
        return D/W

# ============================================================================
def wmean_2D_serial(x, y, data, l, method='hamming', interp=False):
    """ Temporary solution
    """
    import maud
    return maud.wmean_2D_serial(x, y, data, l, method, interp)

def wmean_2D(x, y, data, l, method='hamming', interp=False):
    """ Temporary solution
    """
    return wmean_2D_serial(x, y, data, l, method, interp)
# ============================================================================
def window_mean_2D_latlon(Lat, Lon, data, l, method='hamming', interp=False):
    """
        Cython version of the window_mean_2D_latlon()

        Input:
          - Lat: 2D array with latitudes
          - Lon: 2D array with longitudes
          - data: An array or a masked array. It must be 2D or 3D. In
	      the case of a 3D, the two last dimensions will be
	      considered as lat and lon, while it will run along the
	      first dimension with the multiprocessing.
          - l: window filter size, in meters
          - method: ['hann', 'hamming', 'blackman']
	      Defines the weight function type

        Output: An array of the same type and dimension of the input
	  data. This will be the low (frequency) pass filtered data,
	  i.e. it will eliminate all short variability

        !!!ATENTION!!!
        - Might be a good idea to eliminate the dependence on
          fluid.
    """

    #if type(data) == dict:
    #    output = {}
    #    for k in data.keys():
    #        output[k] = window_mean_2D_latlon(Lat, Lon, data[k], l, method)
    #    return output

    assert (Lat.ndim == 2) & (Lon.ndim == 2), "Lat and Lon must be 2D array"
    #assert data.ndim == 2, "Sorry, for now I'm only handeling 2D arrays"

    # ==== data is a 2D array ======================================
    if data.ndim == 2:
        if hasattr(data, 'mask'):
            if (data.mask == True).any():
                data_smooth, mask = apply_window_mean_2D_latlon_masked(Lat,
                        Lon, data.data, data.mask.astype('int8'), l, method,
                        interp)
                return ma.masked_array(data_smooth, mask)
            else:
                data_smooth = apply_window_mean_2D_latlon(Lat, Lon, data.data,
                        l, method)
                return ma.array(data_smooth)
        else: # type(data) == np.ndarray:
            return apply_window_mean_2D_latlon(Lat, Lon, data, l, method)

    # ==== data is a 3D array ======================================
    elif data.ndim == 3:
        try:
            from progressbar import ProgressBar
        except:
            print "ProgressBar is not available"

        import multiprocessing as mp
        npes = 2 * mp.cpu_count()
        print " Will work with %s npes" % npes
        data_smooth = ma.empty(data.shape)
        pool = mp.Pool(npes)
        results = []

        N = data.shape[0]

        try:
            pbar = ProgressBar(maxval=N).start()
        except:
            pass

        for n in range(N):
            results.append( pool.apply_async( window_mean_2D_latlon, (Lat, Lon, data[n], l, method, interp) ) )

        for i, r in enumerate(results):
            try:
                pbar.update(i)
            except:
                pass
            data_smooth[i] = r.get()

        return data_smooth

    #if interp == True:
    #    I, J = data.shape
    #    I, J = np.meshgrid(range(I), range(J))
    #    I = I.reshape(-1); J = J.reshape(-1)
    #else:
    #    I, J = np.nonzero(np.isfinite(data))

#        #ind, r = cfind_closer_then(Lat, Lon, Lat[i,j], Lon[i,j], llimit=l)
#        #if len(r) > 0:
#        #    w = weight_func(r, l)
#        #    tmp = data[ind]*w
#        #    wsum = w[np.nonzero(tmp)].sum()
#        #    if wsum != 0:
#        #        data_smooth[i,j] = (tmp).sum()/wsum
#	#elif data.ndim == 3:
#	#    for k in range(data.shape[0]):
#	#        data_smooth[k,i,j] = (data[k][ind]*w).sum()/w[good].sum()
#	#    else:
#	#        data_smooth[k,i,j] = (data[k][ind]*w).sum()/w[good].sum()
#    return data_out
    #return data_smooth

def apply_window_mean_2D_latlon(np.ndarray[DTYPE_t, ndim=2] Lat,
        np.ndarray[DTYPE_t, ndim=2] Lon, np.ndarray[DTYPE_t, ndim=2] data,
        double l, method='hamming'):
    """
    """

    cdef unsigned int i, ii, j, jj
    cdef double r, w
    cdef unsigned int I = data.shape[0]
    cdef unsigned int J = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros((I,J))
    cdef np.ndarray[DTYPE_t, ndim=2] W = np.zeros((I,J))

    weight_func = window_func_scalar(method)

    for i in xrange(I):
        for j in xrange(J):
            for ii in xrange(i, I):
                for jj in xrange(j, J):
                    r = _haversine_scalar(Lat[i,j], Lon[i,j],
                            Lat[ii,jj], Lon[ii,jj])
                    if r <= l:
                        w = weight_func(r, l)
                        if w != 0:
                            D[i, j] += data[ii, jj] * w
                            W[i, j] += w

                            if (i != ii) & (j != jj):
                                D[ii, jj] += data[i, j] * w
                                W[ii, jj] += w

    return D/W


def apply_window_mean_2D_latlon_masked(np.ndarray[DTYPE_t, ndim=2] Lat,
		np.ndarray[DTYPE_t, ndim=2] Lon,
		np.ndarray[DTYPE_t, ndim=2] data,
		np.ndarray[np.int8_t, ndim=2] mask,
		double l, method, interp):
#    """
#    """
    weight_func = window_func_scalar(method)

    cdef unsigned int i, ii, j, jj
    cdef double r, w
    cdef unsigned int I = data.shape[0]
    cdef unsigned int J = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros((I,J))
    cdef np.ndarray[DTYPE_t, ndim=2] W = np.zeros((I,J))
    cdef np.ndarray[DTYPE_t, ndim=2] data_smooth = np.empty((I,J))
    cdef np.ndarray[np.int8_t, ndim=2] mask_smooth = \
		    np.ones((I,J), dtype=np.int8)

    for i in xrange(I):
        for j in xrange(J):
            if (interp is True) or (mask[i, j] == 0):
                for ii in xrange(i, I):
                    for jj in xrange(j, J):
                        if mask[ii, jj] == 0:
                            r = _haversine_scalar(Lat[i,j], Lon[i,j],
                                    Lat[ii,jj], Lon[ii,jj])
                            if r <= l:
                                w = weight_func(r, l)
                                if w != 0:
                                    if mask[ii, jj] == 0:
                                        D[i, j] += data[ii, jj] * w
                                        W[i, j] += w
                                    if mask[i, j] == 0:
                                        D[ii, jj] += data[i, j] * w
                                        W[ii, jj] += w
    for i in xrange(I):
        for j in xrange(J):
            if W[i, j] != 0:
                data_smooth[i, j] = D[i,j]/W[i,j]
                mask_smooth[i, j] = 0

    return data_smooth, mask_smooth
