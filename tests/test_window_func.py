""" Tests to evaluate distance estimate using haversine.
"""

import numpy as np
from numpy.random import random

from maud.window_func import window_func
from maud.cwindow_func import window_func as cwindow_func
from maud.cwindow_func_scalar import window_func_scalar as cwindow_func_scalar


WINTYPES = ['hamming', 'hann', 'blackman', 'boxcar']


def test_load_wintypes():
    for wintype in WINTYPES:
        winfunc = window_func(wintype)
        cwinfunc = cwindow_func_scalar(wintype)


def test_PxC(N=25):
    r = random(N)
    l = 2*random(N) # Wider l, so it is more frequent r<l

    for wintype in WINTYPES:
        print("Testing wintype: %s" % wintype)
        winfunc = window_func(wintype)
        cwinfunc = cwindow_func_scalar(wintype)
        W = winfunc(r, l)
        for i, w in enumerate(W):
            c_W = cwinfunc(r[i], l[i])
            assert(c_W == W[i])
