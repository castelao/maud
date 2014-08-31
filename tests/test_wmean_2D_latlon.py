""" Test some fundamental results from  wmean_2D_latlon
"""

import numpy as np
from numpy import ma
from numpy.random import random

from maud import tests_support
from maud import wmean_2D_latlon_serial, wmean_2D_latlon
from cmaud import wmean_2D_latlon as cwmean_2D_latlon


#def random_input(N=10):
#    I, J = (N*random(2)).astype('i')+1



WINTYPES = ['hamming', 'hann', 'blackman', 'boxcar']


def test_inputsizes():
    tests_support.inputsizes_f2D(wmean_2D_latlon)
    tests_support.inputsizes_f2D(cwmean_2D_latlon)


def test_mask(N=4):
    tests_support.masked_input_2D(wmean_2D_latlon, N)
    tests_support.masked_input_2D(cwmean_2D_latlon, N)


def test_whitenoise():
    """
        Apply in a 3D array.

        Need to improve this.
    """
    grid = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(grid, grid)

    #h = ma.array(random(X.shape)-0.5)
    h = ma.array(random([3]+list(X.shape))-0.5)

    smooth1 = wmean_2D_latlon(X, Y, h, l=700e3)
    #y2 = cmaud.window_1Dmean(Z, l=l, axis=2, method='hamming')

    # Large limits since the filter does not include too many numbers
    assert abs(smooth1).mean() < 0.05
    assert abs(smooth1).max() < 0.1


def test_2Dmasked_array(N=25):
    l = N/2

    # Ones array
    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = random((N, N))
    thr = np.percentile(data, 70)
    data = ma.masked_greater(data, thr)
    h = wmean_2D_latlon(X, Y, data, l=l)
    assert h.mask.any()


def test_ones(N=9):
    """ The energy must be preserved

        Therefore, an array of ones must return only ones, even if
          the input has mask, and with interp.
    """
    l = N/2

    print("Testing 2D array")
    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = np.ones((N, N))
    tests_support.eval_ones(wmean_2D_latlon, X, Y, data, l)
    tests_support.eval_ones(cwmean_2D_latlon, X, Y, data, l)

    print("Testing 3D array")
    data = np.ones((3, N, N))
    tests_support.eval_ones(wmean_2D_latlon, X, Y, data, l)
    tests_support.eval_ones(cwmean_2D_latlon, X, Y, data, l)


def test_mask_at_interp():
    """ Test the behavior of masked points with interp on|off

        As long as the filter is wide enough to capture at least
          one data point per point, the interp=True will return
    """
    N = 25
    l = N/2

    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = np.ones((N, N))
    thr = np.percentile(data, 90)
    data = ma.masked_greater(data, thr)
    # Equivalent to interp=False
    h = wmean_2D_latlon(X, Y, data, l=l)
    assert (data.mask == h.mask).all()
    h = wmean_2D_latlon(X, Y, data, l=l, interp=True)
    assert (~h.mask).all()


def test_Serial_x_Parallel(N=10):
    """

        Improve this. Should include more possibilities like:
          different arrays shapes, l, input types(array x MA)
    """
    l = N/2

    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = random(X.shape)
    h_serial = wmean_2D_latlon_serial(X, Y, data, l=l)
    h = wmean_2D_latlon(X, Y, data, l=l)
    assert (h_serial == h).all()


def test_Python_x_Cython(N=10):
    l = N/2
    # ATENTION, in the future I should not force t to be np.float.
    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = random(X.shape)
    h = wmean_2D_latlon(X, Y, data, l=l)
    ch = cwmean_2D_latlon(X, Y, data, l=l)
    assert (h == ch).all()




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
