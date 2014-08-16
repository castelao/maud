# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4 expandtab

import numpy as np
from numpy import ma

import multiprocessing as mp

cimport numpy as np
from libc.math cimport cos

from maud.cwindow_func import window_func, window_func_scalar

try:
    from progressbar import ProgressBar
except:
    print "ProgressBar is not available"


np.import_array()

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef double DEG2NM  = 60.
cdef double NM2M   = 1852.    # Defined in Pond & Pickard p303.
cdef double DEG2M = DEG2NM*NM2M
cdef double deg2rad = 0.5*np.pi/180


def cdistance_scalar(double lat, double lon, double lat_c, double lon_c):
    """
    """
    #assert lat.dtype == DTYPE and  lon.dtype == DTYPE
    return ((lat-lat_c)**2 +
                    ((lon-lon_c) * cos((lat_c+lat)*deg2rad))**2)**.5 * DEG2M


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

def apply_window_1Dmean(np.ndarray[DTYPE_t, ndim=1] data, double t0, np.ndarray[DTYPE_t, ndim=1] t, double l, np.ndarray[np.int_t, ndim=1] I, weight_func):
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
            if (data.mask==True).any():
                data_smooth, mask = apply_window_mean_2D_latlon_masked(Lat, Lon,
				data.data, data.mask.astype('int8'), l, method,
				interp)
                return ma.masked_array(data_smooth, mask)
            else:
                data_smooth = apply_window_mean_2D_latlon(Lat, Lon, data.data, l,
				method, interp)
                return ma.array(data_smooth)
        else: # type(data) == np.ndarray:
            return apply_window_mean_2D_latlon(Lat, Lon, data, l, method, interp)

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

def apply_window_mean_2D_latlon(np.ndarray[DTYPE_t, ndim=2] Lat, np.ndarray[DTYPE_t, ndim=2] Lon, np.ndarray[DTYPE_t, ndim=2] data, l, method='hamming'):
#    """
#    """
    weight_func = window_func_scalar(method)

    cdef unsigned int i, ii, j, jj
    cdef double r, w
    cdef double W, D
    cdef unsigned int I = data.shape[0]
    cdef unsigned int J = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] data_smooth = np.empty((I,J))

    for i in range(I):
        for j in range(J):
            W = 0
            D = 0
            for ii in range(I):
                for jj in range(J):
                    r = cdistance_scalar(Lat[i,j], Lon[i,j],
				    Lat[ii,jj], Lon[ii,jj])
                    if r <= l:
                        w = weight_func(r, l)
                        if w != 0:
                            D += data[ii, jj] * w
                            W += w
            if W != 0:
                data_smooth[i, j] = D/W

    return data_smooth

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
    cdef double W, D
    cdef unsigned int I = data.shape[0]
    cdef unsigned int J = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] data_smooth = np.empty((I,J))
    cdef np.ndarray[np.int8_t, ndim=2] mask_smooth = \
		    np.ones((I,J), dtype=np.int8)

    for i in range(I):
        for j in range(J):
            if (interp is True) or (mask[i, j] == 0):
                W = 0
                D = 0
                for ii in range(I):
                    for jj in range(J):
                        if mask[ii, jj] == 0:
                            r = cdistance_scalar(Lat[i,j], Lon[i,j],
					    Lat[ii,jj], Lon[ii,jj])
                            if r <= l:
                                w = weight_func(r, l)
                                if w != 0:
                                    D += data[ii, jj] * w
                                    W += w
                if W != 0:
                    data_smooth[i, j] = D/W
                    mask_smooth[i, j] = 0

    return data_smooth, mask_smooth
