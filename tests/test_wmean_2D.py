""" Test some fundamental results from  window_mean_2D_latlon
"""

import numpy as np
from numpy import ma
from numpy.random import random
from maud import wmean_2D

#def random_input(N=10):
#    I, J = (N*random(2)).astype('i')+1

def test_whitenoise():
    """
        Apply in a 3D array.

        Need to improve this.
    """
    grid = np.arange(-10,10,0.25)
    X, Y = np.meshgrid(grid, grid)

    #h = ma.array(random(X.shape)-0.5)
    h = ma.array(random([3]+list(X.shape))-0.5)

    smooth1 = wmean_2D(X, Y, h, l=7.8)
    #y2 = cmaud.window_1Dmean(Z, l=l, axis=2, method='hamming')

    # Large limits since the filter does not include too many numbers
    assert abs(smooth1).mean() < 0.05
    assert abs(smooth1).max() < 0.1
