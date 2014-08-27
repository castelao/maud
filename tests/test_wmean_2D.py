""" Test some fundamental results from  window_mean_2D_latlon
"""

import numpy as np
from numpy import ma
from numpy.random import random
from maud import wmean_2D, wmean_2D_serial
from cmaud import wmean_2D as cwmean_2D
from cmaud import wmean_2D_serial as cwmean_2D_serial

#def random_input(N=10):
#    I, J = (N*random(2)).astype('i')+1

def test_inputsizes():
    l = 3

    # 1D input
    #x = np.arange(10)
    #y = x
    #z = random(x.shape)
    #h = wmean_2D(x, y, z, l)

    # 2D input
    x = np.arange(10)
    y = np.arange(3)
    X, Y = np.meshgrid(x, y)
    Z = random(X.shape)
    h = wmean_2D(X, Y, Z, l)
    assert Z.shape == h.shape

    # 3D input
    Z = random([3]+list(X.shape))
    h = wmean_2D(X, Y, Z, l)
    assert Z.shape == h.shape


def test_mask(N=4):
    l = 5

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # input ndarray -> output ndarray
    Z = np.ones(X.shape)
    h = wmean_2D(X, Y, Z, l=l)
    assert type(h) is np.ndarray

    # input MA array -> output MA array
    Z = ma.array(Z)
    h = wmean_2D(X, Y, Z, l=l)
    assert type(h) == ma.MaskedArray
    # Input MA and mask==False -> Output MA and mask==False
    assert ~h.mask.any()

    # Only the masked inputs should return as masked.
    Z.mask = ma.getmaskarray(Z)
    Z.mask[0, 0] = True
    h = wmean_2D(X, Y, Z, l=l)
    assert h[0, 0].mask == True
    assert ~h[1:, 1:].mask.any()

def test_whitenoise():
    """
        Apply in a 3D array.

        Need to improve this.
    """
    grid = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(grid, grid)

    #h = ma.array(random(X.shape)-0.5)
    h = ma.array(random([3]+list(X.shape))-0.5)

    smooth1 = wmean_2D(X, Y, h, l=7.8)
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
    h = wmean_2D(X, Y, data, l=l)
    assert h.mask.any()

def eval_ones(x, y, z, l):

    h = wmean_2D(x, y, z, l=l)
    assert (h == 1).all()

    # Ones masked array with random masked positions
    tmp = random(z.shape)
    # Select the top 1 third of the positions
    thr = np.percentile(tmp, 70)

    z = ma.masked_array(z, tmp>=thr)
    h = wmean_2D(x, y, z, l=l)
    assert (h == 1).all()

    # Masked values should not interfere in the filtered output.
    z.data[z.mask==True] = 1e10
    h = wmean_2D(x, y, z, l=l)
    assert (h == 1).all()

    # With interp, the energy should also be preserved
    h = wmean_2D(x, y, z, l=l, interp=True)
    assert (h == 1).all()

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
    eval_ones(X, Y, data, l)

    print("Testing 3D array")
    data = np.ones((3, N, N))
    eval_ones(X, Y, data, l)


def test_Serial_x_Parallel(N=10):
    """

        Improve this. Should include more possibilities like:
          different arrays shapes, l, input types(array x MA)
    """
    l = N/2

    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = random(X.shape)
    h_serial = wmean_2D_serial(X, Y, data, l=l)
    h = wmean_2D(X, Y, data, l=l)
    assert (h_serial == h).all()


def test_Python_x_Cython(N=10):
    l = N/2
    # ATENTION, in the future I should not force t to be np.float.
    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = random(X.shape)
    h = wmean_2D(X, Y, data, l=l)
    ch = cwmean_2D(X, Y, data, l=l)
    assert (h == ch).all()
