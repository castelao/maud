from maud.cwindow_func import window_func

import numpy as np
from numpy import ma

cimport numpy as np
from libc.math cimport cos

#from fluid.common.distance import distance
from fluid.common.distance import find_closer_then
#from fluid.cdistance import distance
#from fluid.cdistance import find_closer_then


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


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
        t = np.arange(data.shape[axis])

    # t must has the same shape of data along axis
    assert t.shape == (data.shape[axis],)

    # ----
    winfunc = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    cdef int i

    if data.ndim==1:
        # It's faster than getmaskarray
        (I,) = np.nonzero(np.isfinite(data))
        for i in I:
            dt = t-t[i]
            ind = np.nonzero((np.absolute(dt)<l))
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

    if type(data) == dict:
        #print "Sorry, I'm not ready yet to handle dictionaries. Please, run variable by variable."
        return

    assert (Lat.ndim == 2) & (Lon.ndim == 2), "Lat and Lon must be 2D array"
    assert data.ndim == 2, "data must be a 2D array"

    weight_func = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    I, J = np.nonzero(np.isfinite(data))
    #for i in range(I):
    #    for j in range(J):
    for i, j in zip(I,J):
            ind, r = find_closer_then(Lat, Lon, Lat[i,j], Lon[i,j], llimit=l)

            w = weight_func(r, l)
            if len(data.shape)==2:
                good = np.nonzero(data[ind])
                #ind = np.nonzero(data[key])
                # Stupid solution!!! Think about a better way to do this.
                if not hasattr(data[i,j],'mask'):
                    data_smooth[i,j] = (data[ind]*w).sum()/w[good].sum()
                else:
                    if data[i,j].mask==False:
                        data_smooth[i,j] = (data[ind]*w).sum()/w[good].sum()
            elif len(data.shape)==3:
                for k in range(data.shape[0]):
                    if not hasattr(data[k,i,j],'mask'):
                        data_smooth[k,i,j] = (data[k][ind]*w).sum()/w[good].sum()
                    else:
                        if data.mask[k,i,j]==False:
                            data_smooth[k,i,j] = (data[k][ind]*w).sum()/w[good].sum()
    return data_smooth
