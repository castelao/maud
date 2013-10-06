from maud.cwindow_func import window_func

import numpy as np
from numpy import ma

cimport numpy as np
from libc.math cimport cos

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
