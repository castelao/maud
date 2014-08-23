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

cdef double DEG2RAD = M_PI/180.
cdef float AVG_EARTH_RADIUS = 6371000  # in m



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
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))

    return h


def haversine_scalar(lat, lon, lat_c, lon_c):
    """ Python wrapper for _haversine_scalar()
    """
    return _haversine_scalar(lat, lon, lat_c, lon_c)


def haversine(lat, lon, lat_c, lon_c):
    """ Python wrapper for _haversine_scalar()
    """
    if (type(lat) == float) & (type(lon) == float):
        return _haversine_scalar(lat, lon, lat_c, lon_c)

    elif hasattr(lat, 'mask'):
        L = np.empty(lat.shape)
  
    return _haversine_scalar(lat, lon, lat_c, lon_c)
