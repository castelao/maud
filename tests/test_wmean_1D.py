""" Test some fundamental results from  window_mean_2D_latlon
"""

import numpy as np
from numpy import ma
from numpy import pi
from numpy.random import random
import maud

try:
    import cython
    with_cython = True
except:
    with_cython = False

if with_cython:
    import cmaud

def test_mask(N=10):
    l = 5
    t = np.arange(N)
    y = np.ones(N)

    # Input ndarray -> output ndarray
    h = maud.wmean_1D(y, t=t, l=l)
    assert type(h) is np.ndarray
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l)
        assert type(h) is np.ndarray

    y = ma.array(y)
    h = maud.wmean_1D(y, t=t, l=l)
    # Input MA -> output MA
    assert type(h) == ma.MaskedArray
    # Input MA and mask==False -> Output MA and mask==False
    assert ~h.mask.any()
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l)
        assert type(h) == ma.MaskedArray
        assert ~h.mask.any()

    y.mask = ma.getmaskarray(y)
    y.mask[0] = True
    h = maud.wmean_1D(y, t=t, l=l)
    # The masked values @ input will be masked @ output
    assert h[0].mask == True
    assert ~h[1:].mask.any()
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l)
        assert h[0].mask == True
        assert ~h[1:].mask.any()


def eval_ones(y, t, l):

    h = maud.wmean_1D(y, t=t, l=l)
    assert (h == 1).all()
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l)
        assert (h == 1).all()

    # Ones masked array with random masked positions
    tmp = random(y.shape)
    # Select the top 1 third of the positions
    thr = np.percentile(tmp, 70)

    y = ma.masked_array(y, tmp>=thr)
    h = maud.wmean_1D(y, t=t, l=l)
    assert (h == 1).all()
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l)
        assert (h == 1).all()

    # Masked values should not interfere in the filtered output.
    y.data[y.mask==True] = 1e10
    h = maud.wmean_1D(y, t=t, l=l)
    assert (h == 1).all()
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l)
        assert (h == 1).all()

    # With interp, the energy should also be preserved
    h = maud.wmean_1D(y, t=t, l=l, interp=True)
    assert (h == 1).all()
    if with_cython:
        h = cmaud.wmean_1D(y, t=t, l=l, interp=True)
        assert (h == 1).all()

def test_ones(N=25):
    """ The energy must be preserved

        Therefore, an array of ones must return only ones, even if
          the input has mask, and with interp.
    """
    l = N/2

    print("Testing 1D array")
    t = np.arange(N)
    y = np.ones(N)
    eval_ones(y, t, l)

    #print("Testing 2D array")
    t = np.arange(N)
    y = np.ones((N,3))
    eval_ones(y, t, l)


def test_mask_at_interp():
    """ Test the behavior of masked points with interp on|off

        As long as the filter is wide enough to capture at least
          one data point per point, the interp=True will return
    """
    N = 25
    t = np.arange(N)
    y = random(N)
    yn = y[y.argsort()[-5]]
    y = ma.masked_greater(y, yn)
    # Equivalent to interp=False
    h = maud.wmean_1D(y, t=t, l=11)
    assert (y.mask == h.mask).all()
    h = maud.wmean_1D(y, t=t, l=5, interp=True)
    assert (~h.mask).all()


def test_Serial_x_Parallel(N=10):
    """

        Improve this. Should include more possibilities like:
          different arrays shapes, l, input types(array x MA)
    """
    l = 5
    t = np.arange(N)
    y = random(N)
    h_serial = maud.wmean_1D_serial(y, t=t, l=l)
    h = maud.wmean_1D(y, t=t, l=l)
    assert (h_serial == h).all()

def Python_x_Cython(N=10):
    if not with_cython:
        return

    l = 5
    # ATENTION, in the future I should not force t to be np.float.
    t = np.arange(N, dtype=np.float)
    y = random((N, 3))
    h = maud.wmean_1D_serial(y, t=t, l=l)
    ch = cmaud.wmean_1D(y, t=t, l=l)
    assert (h == ch).all()


def test_answer():
    """

        Consider to sum a reference function to make the test harder,
          otherwise I'm just filtering white noise.
    """
    if not with_cython:
        return


    N = 7
    Nt = 20
    l = 10
    #x = np.arange(N)/5.
    #X, Y = np.meshgrid(x, x)
    Z = ma.array(np.random.random((N, N, Nt)))

    y1 = maud.wmean_1D(Z, l=l, axis=2, method='hamming')
    y2 = cmaud.window_1Dmean(Z, l=l, axis=2, method='hamming')

    err = y1 - y2
    assert abs(err).mean() < 1e-10
    assert abs(err).var() < 1e-10
    assert abs(err).max() < 1e-10


def test_pass_all():
    """

    """

    T1 = 50*np.random.random()
    l = T1/50
    t = np.arange(-50,50.5,.5)
    Z = ma.array(np.random.random(t.size))
    S1 = np.sin(2*pi*t/T1) + 10
    y = S1 + Z
    h = maud.wmean_1D(y, l=l, axis=0, method='hamming')
    err = y - h
    assert abs(err).mean() < 1e-10


def test_sin_diff():
    """

    """

    T1 = 100
    l = T1/2
    t = np.arange(-50,50.5,.5)
    Z = ma.array(np.random.random(t.size))
    S1 = np.sin(2*pi*t/T1)
    y = S1 + Z
    h = maud.wmean_1D(y, l=l, axis=0, method='hamming')
    err = S1 - h
    assert (abs(err).mean() < Z).all


def notest_bandpass():
    """

        Creates another test with random window lenghts
    """
    N = 100
    x =  ma.array(np.sort(N*np.random.random(N)))
    y1 = 3*np.sin(2*pi * x/(N/3.) + 2*pi*np.random.random())
    y2 = 3*np.sin(2*pi * x/(N/11.) + 2*pi*np.random.random())
    y3 = 3*np.sin(2*pi * x/(N/21.) + 2*pi*np.random.random())
    y = y1 + y2 + y3 + 1*(np.random.randn(x.size))

    f = y - maud.wmean_1D(data=y, l=N/7., t=x)
    f = maud.wmean_1D(data=f, l=N/15., t=x)
    fb = maud.wmean_bandpass_1D(data=y, lshorterpass=N/15., llongerpass=N/7.,
            t=x)
    assert ma.allclose(f, fb)

