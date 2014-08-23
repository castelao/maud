"""
"""

import numpy as np
from numpy import radians


AVG_EARTH_RADIUS = 6371000  # in m


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

def haversine(lat, lon, lat_c, lon_c):

    lat, lon, lat_c, lon_c = map(radians, [lat, lon, lat_c, lon_c])

    dlat = lat - lat_c
    dlon = lon - lon_c
    d = np.sin(dlat / 2) ** 2 + \
            np.cos(lat) * np.cos(lat_c) * np.sin(dlon / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h
