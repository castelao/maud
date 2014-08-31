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


def eval_ones(f, x, y, z, l):

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

