""" Tests to evaluate distance estimate using haversine.
"""

import numpy as np
from numpy.random import random
from maud.distance import haversine


try:
    import cython
    with_cython = True
except:
    with_cython = False

if with_cython:
    from maud.cdistance import haversine as c_haversine
    from maud.cdistance import haversine_scalar

def test_zerodistance(N=25):
    """ Distance from one point to itself
    """
    for n in range(N):
        lon = 540*random() - 180
        lat = 180*random() - 90
        assert (haversine(lon, lat, lon, lat) == 0.0)
        
        if with_cython:
            #assert (c_haversine(lon, lat, lon, lat) == 0.0)
            assert (haversine_scalar(lon, lat, lon, lat) == 0.0)


def test_360offset(N=25):
    """ Distance from itself around a full circle
    """
    X = -180 * random(N) # N values from [-180, 0]
    X = np.append([-180, 0], X) # Add the two limits -180 and 0
    Y = 90*(2*random(N)-1)
    Y = np.append([-90, 0, 90], Y)
    for x, y in zip(X, Y):
        d = haversine(x, y, x+360, y)
        assert (d<1e-3) # Less then a milimiter

        if with_cython:
            c_d = haversine_scalar(x, y, x+360, y)
            assert (c_d<1e-3) # Less then a milimiter


def test_PxC(N=25):
    """ Python and Cython should give the same answer
    """
    if not with_cython:
        return

    lon0 = 10
    lat0 = -25
    Lon = 540*random(N) - 180
    Lat = 90*(2*random(N)-1)

    Lon, Lat = np.meshgrid(Lon, Lat)

    #print("Testing scalar")
    #d = haversine(Lat[0, 0], Lon[0, 0], lat0, lon0)
    #c_d = c_haversine(Lat[0, 0], Lon[0, 0], lat0, lon0)
    #assert abs(d - c_d).max() < 1e-6

    print("Testing 1D array")
    for n in range(N):
        d = haversine(Lat[n], Lon[n], lat0, lon0)
        c_d = c_haversine(Lat[n], Lon[n], lat0, lon0)
        assert abs(d - c_d).max() < 1e-6

    print("Testing 2D array")
    d = haversine(Lat, Lon, lat0, lon0)
    c_d = c_haversine(Lat, Lon, lat0, lon0)
    assert abs(d - c_d).max() < 1e-6


#def test_haversine_scalar(N=25)
#
#    for lon, lat in zip(Lon, Lat):
#        d = haversine(lat, lon, lat0, lon0)
#        c_d = haversine_scalar(lat, lon, lat0, lon0)
#        # assert returns np.float64, while haversine_scalar returns float
#        # Should they return the same type? So which one?
#        #assert type(d) == type(c_d)
#        assert abs(d - c_d) < 1e-6


def  test_onedegreedistance(N=25):
    """
    """
    Lon, Lat = random((2, N))
    Lon = 540*Lon- 180
    Lat = 89.5*(2*Lat-1)
    for lon, lat in zip(Lon, Lat):
        d = haversine(lat-0.5, lon, lat+0.5, lon)
        assert abs(d-111195)<0.1

        if with_cython:
            c_d = haversine_scalar(lat-0.5, lon, lat+0.5, lon)
            assert abs(c_d-111195)<0.1

