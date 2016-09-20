from numpy import array, absolute
from numpy.random import random

try:
    import cython
    with_cython = True
except:
    with_cython = False

from maud.window_func import _weight_hamming
if with_cython:
    from maud.cwindow_func import _weight_hamming as c_weight_hamming
    from maud.cwindow_func_scalar import _weight_hamming_scalar


def test_knownanswer():
    r = array([-3, -2, -1, 0, 1, 2, 3, 100])
    l = 5
    w = _weight_hamming(r,l)
    d = w - array([ 0., 0.16785218, 0.68214782, 1., 0.68214782, 0.16785218, 0., 0.])
    assert absolute(d).max() < 1e-8


def test_PxC(N=50):
    if not with_cython:
        return

    for n in range(N):
        r = 5*(2*random(10)-1)
        l = 10*random()
        w = _weight_hamming(r,l)
        cw = c_weight_hamming(r,l)
        assert type(w) == type(cw)
        assert absolute(w - cw).max() < 1e-10


def test_cython_scalar():
    if not with_cython:
        return

    R = array([-3, -2, -1, 0, 1, 2, 3, 100])
    W = array([ 0., 0.16785218, 0.68214782, 1., 0.68214782, 0.16785218, 0., 0.])
    l = 5
    for r, w in zip(R, W):
        w2 = _weight_hamming_scalar(r, l)
        d = w - w2
        assert absolute(d) < 1e-8


def out_of_window():
    r = 5*(2*random(10)-1)
    l = 10*random()
    w = _weight_hamming(r,l)
    ind = r>l/2
    assert (w[ind]==0).all()
    if with_cython:
        cw = c_weight_hamming(r,l)
        assert (cw[ind]==0).all()


# Question: _weight_hann(ma.masked_all(3), 5) should return a masked array?
# _weight_hann(ma.masked_all(4),5)
