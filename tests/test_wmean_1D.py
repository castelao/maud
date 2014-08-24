""" Test some fundamental results from  window_mean_2D_latlon
"""

import numpy as np
from numpy import ma
from numpy import pi
import maud
import cmaud


def test_answer():
    """

        Consider to sum a reference function to make the test harder,
          otherwise I'm just filtering white noise.
    """

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


def test_pass_all():
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
