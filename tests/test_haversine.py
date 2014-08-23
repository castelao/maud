""" Tests to evaluate distance estimate using haversine.
"""

import numpy as np
from numpy.random import random
from maud.distance import haversine
from maud.cdistance import haversine_scalar

def test_zerodistance():
    """ Distance from one point to itself
    """
    x, y = random(2)
    assert (haversine(x, y, x, y) == 0.0)
    assert (haversine_scalar(x, y, x, y) == 0.0)


def test_360(N=25):
    """ Distance from itself around a full circle
    """
    Y = 90*(2*random(N)-1)
    Y = np.append([-90, 0, 90], Y)
    for y in Y:
        d = haversine(0, y, 360, y)
        c_d = haversine_scalar(0, y, 360, y)
        assert (d<1e-3) # Less then a milimiter
        assert (c_d<1e-3) # Less then a milimiter


def test_PxC(N=500):
    """ Python and Cython should give the same answer
    """
    lon0 = 10
    lat0 = -25
    Lon = 400*(2*random(N)-1)
    Lat = 90*(2*random(N)-1)

    for lon, lat in zip(Lon, Lat):
        d = haversine(lat, lon, lat0, lon0)
        c_d = haversine_scalar(lat, lon, lat0, lon0)
        # assert returns np.float64, while haversine_scalar returns float
        # Should they return the same type? So which one?
        #assert type(d) == type(c_d)
        assert abs(d - c_d) < 1e-6


def  test_onedegreedistance(N=25):
    """
    """
    Lon, Lat = random((2, N))
    Lon = 400*(2*Lon-1)
    Lat = 89*(2*Lat-1)
    for lon, lat in zip(Lon, Lat):
        d = haversine(lat-0.5, lon, lat+0.5, lon)
        c_d = haversine_scalar(lat-0.5, lon, lat+0.5, lon)

        assert abs(d-111195)<0.1
        assert abs(c_d-111195)<0.1

