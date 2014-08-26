# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
from window_func import window_func


def wmean_2D_serial(x, y, data, l, method='hamming', interp=False):
    """
        - interp
        - split in two solutions, array and masked array
        - Allow nD input arrays
    """
    assert data.ndim >= 2, "The input data must has at leas 2 dimensions"
    assert (x.shape == y.shape), "x and y must have the same shape."
    assert (data.shape[-2:] == x.shape), \
            "The last 2 dimensions of data mush be equal to x & y shape"
    assert type(data) in [np.ndarray, ma.MaskedArray], \
            "data must be an array or masked_array"

    if type(data) is np.ndarray:
        data_smooth = np.empty(data.shape)
    else:
        data_smooth = ma.masked_all(data.shape)

    # ----
    if data.ndim > 2:
        for i in xrange(data.shape[0]):
            data_smooth[i] = wmean_2D_serial(x, y, data[i], l, method, interp)
        return data_smooth
    # Below here it is expected only 2D arrays
    # ----
    winfunc = window_func(method)

    #I, J = data.shape[-2:]
    #for i in xrange(I):
    #    for j in xrange(J):
    if interp == True:
        I, J = data.shape
        I, J = np.meshgrid(range(I), range(J))
        I = I.reshape(-1); J = J.reshape(-1)
    else:
        I, J = np.nonzero(~ma.getmaskarray(data))

    for i, j in zip(I, J):
        data_smooth[..., i, j] = _convolve_2D(x[i,j], y[i,j], x, y, l,
                winfunc, data)

    ind_nan = np.isnan(data_smooth)
    if ind_nan.any():
        data_smooth.mask[ind_nan] = True

    return data_smooth


def _convolve_2D(x0, y0, x, y, l, winfunc, data):
    r = ( (x-x0)**2 + (y-y0)**2 )**0.5
    if len(r) > 0:
        # Index only the valid data that is inside the window
        #ind = (np.absolute(r) < l) & (~ma.getmaskarray(data))
        #w = winfunc(r[ind], l)
        w = winfunc(r, l)
        return _apply_convolve_2D(data, w)


def _apply_convolve_2D(data, w):
    """ Apply weights w into data

        This functions is usefull for arrays with more than 2D, so that the
          r and w are estimated the minimum ammount of times. It assumes that
          the weights (w) are applicable at the last 2 dimensions, and repeat
          the procedure to any number of previous dimensions. I.e., a 2D
          array is straight forward, while a 3D array, the procedure is
          repeated along the first dimension as n 2D arrays.
    """
    if data.ndim > 2:
        output = ma.masked_all(data.shape[:-2])
        I = data.shape[0]
        for i in xrange(I):
            output[i] = _apply_convolve_2D(data[i], w)
        return output

    ind = (w != 0) & (~ma.getmaskarray(data))
    tmp = data[ind]*w[ind]
    wsum = w[ind].sum()
    if wsum != 0:
        return (tmp).sum()/wsum


def wmean_2D(x, y, data, l, method='hamming', interp=False):

    print("I'm not ready yet to run in parallel. I'll do serial instead.")
    return wmean_2D_serial(x, y, data, l, method, interp)
