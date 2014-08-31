import numpy as np
from numpy import ma
from numpy.random import random


def inputsizes_f2D(f):
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
    h = f(X, Y, Z, l)
    assert Z.shape == h.shape

    # 3D input
    Z = random([3]+list(X.shape))
    h = f(X, Y, Z, l)
    assert Z.shape == h.shape


def masked_input_2D(f, N=4):
    l = 2*N/3

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # input ndarray -> output ndarray
    Z = np.ones(X.shape)
    h = f(X, Y, Z, l=l)
    assert type(h) is np.ndarray

    # input MA array -> output MA array
    Z = ma.array(Z)
    h = f(X, Y, Z, l=l)
    assert type(h) == ma.MaskedArray
    # Input MA and mask==False -> Output MA and mask==False
    assert ~h.mask.any()

    # Only the masked inputs should return as masked.
    Z.mask = ma.getmaskarray(Z)
    Z.mask[0, 0] = True
    h = f(X, Y, Z, l=l)
    assert h[0, 0].mask == True
    assert ~h[1:, 1:].mask.any()


def eval_ones_2D(f, x, y, z, l):

    h = f(x, y, z, l=l)
    assert (h == 1).all()

    # Ones masked array with random masked positions
    tmp = random(z.shape)
    # Select the top 1 third of the positions
    thr = np.percentile(tmp, 70)

    z = ma.masked_array(z, tmp>=thr)
    h = f(x, y, z, l=l)
    assert (h == 1).all()

    # Masked values should not interfere in the filtered output.
    z.data[z.mask==True] = 1e10
    h = f(x, y, z, l=l)
    assert (h == 1).all()

    # With interp, the energy should also be preserved
    h = f(x, y, z, l=l, interp=True)
    assert (h == 1).all()


def mask_at_interp(f):
    """ Test the behavior of masked points with interp on|off

        As long as the filter is wide enough to capture at least
          one data point per point, the interp=True will return
    """
    N = 25
    l = 2*N/3

    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = np.ones((N, N))
    thr = np.percentile(data, 90)
    data = ma.masked_greater(data, thr)
    # Equivalent to interp=False
    h = f(X, Y, data, l=l)
    assert (data.mask == h.mask).all()
    h = f(X, Y, data, l=l, interp=True)
    assert (~h.mask).all()


def compare2func(f1, f2):
    N = 10
    l = N/2
    grid = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(grid, grid)
    data = random(X.shape)
    h1 = f1(X, Y, data, l=l)
    h2 = f2(X, Y, data, l=l)
    assert (h1 == h2).all()

    data = ma.array(data)
    h = f1(X, Y, data, l=l)
    h2 = f2(X, Y, data, l=l)
    assert (h1 == h2).all()

    thr = np.percentile(data, 70)
    data = ma.masked_greater(data, thr)
    h = f1(X, Y, data, l=l)
    h2 = f2(X, Y, data, l=l)
    assert (h1 == h2).all()
