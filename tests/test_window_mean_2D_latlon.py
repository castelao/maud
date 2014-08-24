""" Test some fundamental results from  wmean_2D_latlon
"""

import numpy as np
from numpy import ma
from numpy.random import random
from maud import wmean_2D_latlon
from cmaud import window_mean_2D_latlon as cwindow_mean_2D_latlon


WINTYPES = ['hamming', 'hann', 'blackman', 'boxcar']

def latlon_2D(I=10, J=10):
    """ Creates lat,lon positions for tests
    """
    Lon = 560 * random((I, J)) - 180
    Lat = 180 * random((I, J)) - 90
    return Lat, Lon


def test_allmasked(N=10):
    """ If the input is all masked, the output must be all masked
    """
    I = (99 * random(N)).astype('i') + 1
    J = (99 * random(N)).astype('i') + 1
    for i, j in zip(I, J):
        x = ma.masked_all((i, j))
        Lat, Lon = latlon_2D(i, j)
        h_smooth = wmean_2D_latlon(Lat, Lon, x, l=1e10)
        assert h_smooth.mask.all()


def test_ones(N=10):
    """ Test if filter an array of ones, return just ones
    """
    I = (25 * random(N)).astype('i') + 1
    J = (25 * random(N)).astype('i') + 1
    for wintype in WINTYPES:
        print("Testing: %s" % wintype)
        for i, j in zip(I, J):
            h = np.ones((i, j), dtype='f')
            Lat, Lon = latlon_2D(i, j)
            h_smooth = wmean_2D_latlon(Lat, Lon, h, method=wintype, l=1e10)
            assert (h_smooth == h).all()


def whitenoise(Lat, Lon, l):
    h = np.random.random(Lon.shape)-0.5
    h_smooth = wmean_2D_latlon(Lat, Lon, 1+h, method='boxcar', l=l)
    return h_smooth - 1


def hardcoded_maskedarray():
    """Test if masked data is not considered in the average
    """
    h = np.array([[ 1e9,  1e9,  1e9],
         [ 1e9,  3.14,  1e9],
         [ 1e9,  1e9,  1e9]])

    h = ma.masked_greater(h, 10)

    lon = np.array([10.1, 10, 9.9])
    lat = np.array([-0.1, -0.09, -0.08])
    Lon, Lat = np.meshgrid(lon, lat)

    h_smooth = wmean_2D_latlon(Lat, Lon, h, l=1e10)
    h_smooth2 = cwindow_mean_2D_latlon(Lat, Lon, h, l=1e10)
    # maud and cmaud should return the very same result
    assert (h_smooth == h_smooth2).all()
    assert (h_smooth.mask == h.mask).all()
    #assert (h_smooth.compressed() == h.compressed()).all()
    assert (np.absolute(h_smooth - h).sum() == 0.)


def random_maskedarray(N=10, res=0.1):
    #lon0, lat0 = random(2)
    #lon0 = 540*lon0-180 #[180, 360]
    #lat0 = 180*lat0-90

    grid = np.arange(-N/2, N/2)*res
    Lon, Lat = np.meshgrid(grid, grid)

    h = random(Lon.shape)
    h = ma.masked_greater(h, 0.7)

    h_smooth = wmean_2D_latlon(Lat, Lon, h, l=.1)
    h_csmooth = cwindow_mean_2D_latlon(Lat, Lon, h, l=.1)
    assert (h_smooth == h_csmooth).all()

    h_smooth = wmean_2D_latlon(Lat, Lon, h, l=1e10)
    h_csmooth = cwindow_mean_2D_latlon(Lat, Lon, h, l=1e10)
    # maud and cmaud should return the very same result
    assert (h_smooth == h_csmooth).all()
    assert (h_smooth.mask == h.mask).all()
    #assert (h_smooth.compressed() == h.compressed()).all()
    assert (np.absolute(h_smooth - h).sum() == 0.)



def interp():
    """ Test interp option
    """
    lon = np.arange(-1, 10.01, 0.1)
    lat = np.arange(-5, 1.01, 0.1)
    Lon, Lat = np.meshgrid(lon, lat)
    h = ma.masked_greater(np.random.random(Lon.shape), 0.7)
    h_smooth = wmean_2D_latlon(Lat, Lon, h, l=2e5, interp=False)
    h_csmooth = cwindow_mean_2D_latlon(Lat, Lon, h, l=2e5, interp=False)
    h_smooth_i = wmean_2D_latlon(Lat, Lon, h, l=2e5, interp=True)
    h_csmooth_i = cwindow_mean_2D_latlon(Lat, Lon, h, l=2e5, interp=True)
    assert (h_smooth == h_csmooth).all()
    assert (h_smooth_i == h_csmooth_i).all()
    #assert (abs(h_smooth - h_smooth_i).sum() == 0)
    assert ((h_smooth - h_smooth_i) == 0).all()
    assert (h_smooth_i.compressed().size >= h_smooth.compressed().size)

def answer():


    lon = np.arange(-1, 10.01, 0.1)
    lat = np.arange(-5, 1.01, 0.1)
    Lon, Lat = np.meshgrid(lon, lat)

    # ===========================================
    l = 1e6
    err = whitenoise(Lat, Lon, l)
    assert np.absolute(err).mean() < 0.01

    # ===========================================
    #interp()

#np.absolute(err).mean() > 0.02
#
#R = distance(Lon, Lat, 0, -12)
#h = np.sin(2*np.pi*R/1e7)
#noise = 0.1*np.random.random(h.shape)
#
#assert np.absolute(noise).mean() > 0.02
#
#h_smooth = wmean_2D_latlon(Lat, Lon, h, l=1e4)
#err = np.absolute(h_smooth - h).mean()
#
#h_smooth = wmean_2D_latlon(Lat, Lon, (h+noise), l=1e4)
#
#err = np.absolute(h_smooth - h).mean()
#assert err<1e-4
