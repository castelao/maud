
import numpy as np
from numpy import ma

from window_func import window_func
from distance import haversine

 
def wmean_2D_latlon_serial(lat, lon, data, l, method='hamming',
        interp='False'):
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

    """
    #assert ((type(l) == float) or (type(l) == int)), \
    #    "The filter scale (l) must be a float or an int"

    # Temporary solution
    if type(data) == dict:
        output = {}
        for k in data.keys():
            output[k] = wmean_2D_latlon_serial(lat, lon, data[k], l, method,
                    interp)
        return output

    assert data.ndim >= 2, "The input data must has at leas 2 dimensions"
    assert (lat.shape == lon.shape), "lon and lat must have the same shape."
    assert (data.shape[-2:] == lat.shape), \
            "The last 2 dimensions of data mush be equal to lat & lon shape"
    assert type(data) in [np.ndarray, ma.MaskedArray], \
            "data must be an array or masked_array"

    if type(data) is np.ndarray:
        data_smooth = np.empty(data.shape)
    else:
        data_smooth = ma.masked_all(data.shape)

    # ----
    if data.ndim > 2:
        for i in xrange(data.shape[0]):
            data_smooth[i] = wmean_2D_latlon_serial(lat, lon, data[i], l,
                    method, interp)
        return data_smooth
    # Below here it is expected only 2D arrays
    # ----
    winfunc = window_func(method)

    if interp == True:
        I, J = data.shape
        I, J = np.meshgrid(range(I), range(J))
        I = I.reshape(-1); J = J.reshape(-1)
    else:
        I, J = np.nonzero(~ma.getmaskarray(data))

    for i, j in zip(I, J):
        data_smooth[..., i, j] = _convolve_2D_latlon(lat[i,j], lon[i,j],
                lat, lon, l, winfunc, data)

    ind_nan = np.isnan(data_smooth)
    if ind_nan.any():
        data_smooth.mask[ind_nan] = True

    return data_smooth       


def _convolve_2D_latlon(lat0, lon0, lat, lon, l, winfunc, data):
    r = haversine(lat, lon, lat0, lon0)
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
    assert data.shape == w.shape

    if data.ndim > 2:
        output = ma.masked_all(data.shape[:-2])
        I = data.shape[0]
        for i in xrange(I):
            output[i] = _apply_convolve_2D(data[i], w)
        return output

    ind = (w != 0) & (~ma.getmaskarray(data))
    tmp = data[ind]*w[ind]
    # Sum the weights only at the valid data positions.
    wsum = w[ind].sum()
    if wsum != 0:
        return (tmp).sum()/wsum


def wmean_2D_latlon(Lat, Lon, data, l, method='hamming', interp='False'):
    return wmean_2D_latlon_serial(Lat, Lon, data, l, method, interp)
