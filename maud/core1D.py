
"""
"""
# Gui, 27-06-2012
# Just an idea. Create a class of filtered data. A fake Masked Array object
#   which gives the output on demand, filtered.

try:
    import multiprocessing as mp
except:
    print "I couldn't import multiprocessing"

import numpy as np
from numpy import ma

from window_func import window_func
from distance import haversine


def wmean_1D_serial(data, l, t=None, method='hann', axis=0):
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
        data_smooth = wmean_1D_serial(data.swapaxes(0, axis),
            l = l,
            t = t,
            method = method,
            axis = 0)

        return data_smooth.swapaxes(0, axis)
    # Below here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = np.arange(data.shape[axis])

    assert t.shape == (data.shape[axis],), "Invalid size of t."

    # ----
    winfunc = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    if data.ndim==1:
        (I,) = np.nonzero(~ma.getmaskarray(data))
        for i in I:
            data_smooth[i] = _convolve_1D(t[i], t, l, winfunc, data)

    else:
        I = data.shape[1]
        for i in range(I):
            data_smooth[:,i] = wmean_1D_serial(data[:,i],
                    l = l,
                    t = t,
                    method = method,
                    axis = 0)

    return data_smooth


def wmean_1D(data, l, t=None, method='hann', axis=0):
    """ A moving window mean filter, not necessarily a regular grid.

        It is equivalent to wmean_1D_serial but run in parallel with
          multiprocesing for higher efficiency.

        Check wmean_1D_serial documentation for the inputs and other
          details.
    """
    #assert type(data)) in [np.ndarray, ma.MaskedArray]
    assert axis <= data.ndim, "Invalid axis!"

    # If necessary, move the axis to be filtered for the first axis
    if axis != 0:
        data_smooth = wmean_1D(data.swapaxes(0, axis),
            l = l,
            t = t,
            method = method,
            axis = 0)

        return data_smooth.swapaxes(0, axis)
    # Below here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = np.arange(data.shape[axis])

    assert t.shape == (data.shape[axis],), "Invalid size of t."

    # ----
    # Only one dimensions usually means overhead to run in parallel.
    if data.ndim==1:
        data_smooth = wmean_1D_serial(data, l, t=t, method=method, axis=axis)
        return data_smooth
    # ----

    npes = 2 * mp.cpu_count()
    pool = mp.Pool(npes)
    results = []
    I = data.shape[1]
    for i in range(I):
        results.append(pool.apply_async(wmean_1D_serial, \
                (data[:,i], l, t, method, 0)))
    pool.close()

    # Collecting the results.
    data_smooth = ma.masked_all(data.shape)
    for i, r in enumerate(results):
        data_smooth[:,i] = r.get()
    pool.terminate()

    return data_smooth


def _convolve_1D(t0, t, l, winfunc, data):
    """ Effectively apply 1D moving mean along 1D array

        This is not exactly a convolution.

        Support function to wmean_1D

        ATENTION, in the future I should use l/2. in the index for most of
          the weighting windows types.
    """
    dt = t - t0
    # Index only the valid data that is inside the window
    ind = (np.absolute(dt) < l) & (~ma.getmaskarray(data))
    ind = np.nonzero( ind )
    w = winfunc(dt[ind], l)
    return (data[ind] * w).sum() / (w.sum())


def wmean_bandpass_1D_serial(data, lshorterpass, llongerpass, t=None,
        method='hann', axis=0):
    """ Equivalent to wmean_1D_serial, but it is a bandpass

        Input:
            - data: np.array or ma.maked_array, nD
            - lshorterpass: The size of the highpass filter, i.e. shorter
                wavelenghts are preserved. It is in the same unit of t.
            - llongerpass: The size of the lowpass filter, i.e.longer
                wavelenghts are preserved. It is in the same unit of t.
	    - t: is the scale of the choosed axis, 1D. If not
                defined, it will be considered a sequence.
            - method: ['hann', 'hamming', 'blackman']
                Defines the weight function type
            - axis: Dimension which the filter will be applied
    """

    assert axis <= data.ndim, "Invalid axis!"

    # If necessary, move the axis to be filtered for the first axis
    if axis != 0:
        data_smooth = wmean_bandpass_1D_serial(data.swapaxes(0, axis),
                lshorterpass = lshorterpass,
                llongerpass = llongerpass,
                t = t,
                method = method,
                axis = 0)

        return data_smooth.swapaxes(0, axis)
    # Below here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = np.arange(data.shape[axis])

    assert t.shape == (data.shape[axis],), "Invalid size of t."

    # ----
    winfunc = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    if data.ndim==1:
        (I,) = np.nonzero(~ma.getmaskarray(data))
        for i in I:
            # First remove the high frequency
            tmp = _convolve_1D(t[i], t, llongerpass, winfunc, data)
            # Then remove the low frequency
            data_smooth[i] = tmp - \
                    _convolve_1D(t[i], t, lshorterpass, winfunc, tmp)

    else:
        I = data.shape[1]
        for i in range(I):
            data_smooth[:,i] = wmean_bandpass_1D_serial(datai[:,i],
                    lshorterpass, llongerpass, t, method, axis)

    return data_smooth


def wmean_bandpass_1D(data, lshorterpass, llongerpass, t=None,
        method='hann', axis=0):
    """ A bandpass moving window filter, for not necessarily regular grids.

        Equivalent to wmean_bandpass_1D_serial, but it runs in parallel
          with multiprocessing for higher efficiency.

        Check wmean_bandpass_1D_serial documentation for the inputs and
            other details.
    """

    assert axis <= data.ndim, "Invalid axis!"

    # If necessary, move the axis to be filtered for the first axis
    if axis != 0:
        data_smooth = wmean_bandpass_1D(data.swapaxes(0, axis),
                lshorterpass = lshorterpass,
                llongerpass = llongerpass,
                t = t,
                method = method,
                axis = 0)

        return data_smooth.swapaxes(0, axis)
    # Below here, the filter will be always applied on axis=0

    # If t is not given, creates a regularly spaced t
    if t == None:
        print "The scale along the choosed axis weren't defined. I'll consider a constant sequence."
	t = np.arange(data.shape[axis])

    assert t.shape == (data.shape[axis],), "Invalid size of t."

    # ----
    # Only one dimensions usually means overhead to run in parallel.
    if data.ndim==1:
        data_smooth = wmean_bandpass_1D_serial(data, lshorterpass,
                llongerpass, t, method, axis)
        return data_smooth
    # ----

    npes = 2 * mp.cpu_count()
    pool = mp.Pool(npes)
    results = []
    I = data.shape[1]
    for i in range(I):
        results.append(pool.apply_async(wmean_bandpass_1D_serial, \
                (data[:,i], lshorterpass, llongerpass, t, method, axis)))
    pool.close()

    # Collecting the results.
    data_smooth = ma.masked_all(data.shape)
    for i, r in enumerate(results):
        data_smooth[:,i] = r.get()
    pool.terminate()

    return data_smooth
