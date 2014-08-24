# -*- coding: utf-8 -*-
# cython: profile=False

"""

    Wasn't this below supposed to be faster than from i in xrange()?
    #for i from 0 < = i < I:
    #    for j from 0 < = j < J:
"""

import numpy as np

cimport numpy as np
from libc.math cimport cos, sin, asin, sqrt, M_PI


np.import_array()

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef double DEG2RAD = M_PI/180.
cdef float AVG_EARTH_RADIUS = 6371000  # in m
cdef float TWO_AVG_EARTH_RADIUS = 2 * AVG_EARTH_RADIUS # in m



def find_closer_then(lat, lon, lat_c, lon_c, llimit, method="haversine"):
    """
    """
    assert lat.shape == lon.shape, "lat & lon must have same shape"
    #assert (len(lat_c) == 1) & (len(lon_c) == 1)

    deg_limit = np.rad2deg(llimit/AVG_EARTH_RADIUS)
    possible = np.nonzero(abs(lat - lat_c) <= deg_limit)
    L = haversine(lat[possible], lon[possible], lat_c, lon_c)
    ind_closer = L<llimit
    L = L[ind_closer]
    ind = []
    # Repeat for each dimension.
    for p in possible:
        ind.append(p[ind_closer])

    return ind, L


cdef double _haversine_scalar(double lat, double lon, double lat_c,
        double lon_c):
    """ Haversine, gives the grat circle distance in a sphere
    """

    cdef double d
    cdef double h

    lat = lat * DEG2RAD
    lon = lon * DEG2RAD
    lat_c = lat_c * DEG2RAD
    lon_c = lon_c * DEG2RAD

    #sinlat = sin( (lat - lat_c) / 2.)
    #sinlon = sin( (lon - lon_c) / 2.)

    d = sin( (lat - lat_c) / 2.) ** 2 + \
            cos(lat) * cos(lat_c) * sin( (lon - lon_c) / 2) ** 2
    h = TWO_AVG_EARTH_RADIUS * asin(sqrt(d))

    return h


def haversine_scalar(lat, lon, lat_c, lon_c):
    """ Python wrapper for _haversine_scalar()
    """
    return _haversine_scalar(lat, lon, lat_c, lon_c)


def haversine_1Darray(np.ndarray[DTYPE_t, ndim=1] lat,
        np.ndarray[DTYPE_t, ndim=1] lon, double lat_c, double lon_c):
    """ Harversine in a 1D numpy.array

        Inputs:
            lat [1D np.array]: Latitudes
            lon [1D np.array]: Longitudes
            lat_c [float]: Latitude of reference
            lon_c [float]: Longitude of reference
    """

    # Why does this assert fails?
    #assert lat.shape == lon.shape

    cdef unsigned int i
    cdef unsigned int I = lat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] h = np.empty(I)
    cdef double dlat, dlon, d
    cdef double cos_lat_c

    lat_c = lat_c * DEG2RAD
    lon_c = lon_c * DEG2RAD

    cos_lat_c = cos(lat_c)

    for i in xrange(I):
        lat[i] = lat[i] * DEG2RAD
        lon[i] = lon[i] * DEG2RAD

        dlat = lat[i] - lat_c
        dlon = lon[i] - lon_c

        d = sin(dlat / 2) ** 2 + \
            cos(lat[i]) * cos_lat_c * sin(dlon / 2) ** 2
        h[i] = TWO_AVG_EARTH_RADIUS * asin(sqrt(d))

    return h


def haversine_2Darray(np.ndarray[DTYPE_t, ndim=2] lat,
        np.ndarray[DTYPE_t, ndim=2] lon, double lat_c, double lon_c):
    """ Harversine in a 2D numpy.array

        Inputs:
            lat [2D np.array]: Latitudes
            lon [2D np.array]: Longitudes
            lat_c [float]: Latitude of reference
            lon_c [float]: Longitude of reference
    """

    cdef unsigned int i, j
    cdef unsigned int I = lat.shape[0]
    cdef unsigned int J = lat.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] h = np.empty((I, J))
    cdef double dlat, dlon, d
    cdef double cos_lat_c

    lat_c = lat_c * DEG2RAD
    lon_c = lon_c * DEG2RAD

    cos_lat_c = cos(lat_c)

    for i in xrange(I):
        for j in xrange(J):
            lat[i, j] = lat[i, j] * DEG2RAD
            lon[i, j] = lon[i, j] * DEG2RAD

            dlat = lat[i, j] - lat_c
            dlon = lon[i, j] - lon_c

            d = sin(dlat / 2) ** 2 + \
                    cos(lat[i, j]) * cos_lat_c * sin(dlon / 2) ** 2
            h[i, j] = TWO_AVG_EARTH_RADIUS * asin(sqrt(d))

    return h


def haversine(lat, lon, lat_c, lon_c):

    if (type(lat) == np.ndarray) & (type(lon) == np.ndarray):
        if (lat.ndim == 1) & (lon.ndim == 1):
            return haversine_1Darray(lat, lon, lat_c, lon_c)

        if (lat.ndim == 2) & (lon.ndim == 2):
            return haversine_2Darray(lat, lon, lat_c, lon_c)
