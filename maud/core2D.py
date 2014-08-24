# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
from window_func import window_func


def wmean_2D_serial(x, y, data, l, method='hamming'):
    """
        - interp
        - split in two solutions, array and masked array
        - Allow nD input arrays
    """
    assert (x.shape == y.shape), "x and y must have the same shape."
    assert (data.shape[-2:] == x.shape), \
            "The last 2 dimensions of data mush be equal to x & y shape"
    assert type(data) in [np.ndarray, ma.MaskedArray], \
            "data must be an array or masked_array"
    #assert data.ndim == 2, "The input data must be 2D arrays"

    winfunc = window_func(method)

    data_smooth = ma.masked_all(data.shape)

    I, J = data.shape[-2:]
    for i in xrange(I):
        for j in xrange(J):
            data_smooth[..., i, j] = _convolve_2D(x[i,j], y[i,j], x, y, l,
                    winfunc, data)

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
    if data.ndim > 2:
        output = ma.masked_all(data.shape[:-2])
        I = data.shape[0]
        for i in xrange(I):
            output[i] = _apply_convolve_2D(data[i], w)
        return output
    #tmp = data[ind]*w
    tmp = data*w
    wsum = w.sum()
    if wsum != 0:
        return (tmp).sum()/wsum



def wmean_2D(x, y, data, l, method='hamming'):

    print("I'm not ready yet to run in parallel. I'll do serial instead.")
    return wmean_2D_serial(x, y, data, l, method)
