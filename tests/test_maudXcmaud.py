""" Test some fundamental results from  window_mean_2D_latlon
"""

import numpy as np
from numpy import ma
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

    y1 = maud.window_1Dmean(Z, l=l, axis=2, method='hamming')
    y2 = cmaud.window_1Dmean(Z, l=l, axis=2, method='hamming')

    err = y1 - y2
    assert abs(err).mean() < 1e-6
    assert abs(err).var() < 1e-6
    assert abs(err).max() < 1e-6
