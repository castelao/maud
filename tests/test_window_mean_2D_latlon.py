""" Test some fundamental results from  window_mean_2D_latlon
"""

import numpy as np
from numpy import ma
from maud import window_mean_2D_latlon

def whitenoise(Lat, Lon, l):
    h = np.random.random(Lon.shape)-0.5
    h_smooth = window_mean_2D_latlon(Lat, Lon, 1+h, method='boxcar', l=l)
    return h_smooth - 1

def test_answer():

    from fluid.common.distance import distance

    lon = np.arange(-1, 10.01, 0.1)
    lat = np.arange(-5, 1.01, 0.1)
    Lon, Lat = np.meshgrid(lon, lat)

    l = 1e6
    err = whitenoise(Lat, Lon, l)
    assert np.absolute(err).mean() < 0.01


    # Test if masked data is not considered in the average
    h = np.array([[ 1e9,  1e9,  1e9],
         [ 1e9,  3.14,  1e9],
         [ 1e9,  1e9,  1e9]])

    h = ma.masked_greater(h, 10)

    lon = np.array([10.1, 10, 9.9])
    lat = np.array([-0.1, -0.09, -0.08])
    Lon, Lat = np.meshgrid(lon, lat)

    h_smooth = window_mean_2D_latlon(Lat, Lon, h, l=1e10)
    assert (h_smooth.mask == h.mask).all()
    #assert (h_smooth.compressed() == h.compressed()).all()
    assert (np.absolute(h_smooth - h).sum() == 0.)

#np.absolute(err).mean() > 0.02
#
#R = distance(Lon, Lat, 0, -12)
#h = np.sin(2*np.pi*R/1e7)
#noise = 0.1*np.random.random(h.shape)
#
#assert np.absolute(noise).mean() > 0.02
#
#h_smooth = window_mean_2D_latlon(Lat, Lon, h, l=1e4)
#err = np.absolute(h_smooth - h).mean()
#
#h_smooth = window_mean_2D_latlon(Lat, Lon, (h+noise), l=1e4)
#
#err = np.absolute(h_smooth - h).mean()
#assert err<1e-4
