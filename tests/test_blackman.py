from numpy import array, absolute
from numpy.random import random

from maud.window_func import _weight_blackman
#from maud.cwindow_func import _weight_blackman as c_weight_blackman
#from maud.cwindow_func_scalar import _weight_blackman_scalar

# Gotta adjust the coeficcients to the blackman window !!!

def test_knownanswer():
    r = array([-3, -2, -1, 0, 1, 2, 3, 100])
    l = 5
    w = _weight_blackman(r,l)
    d = w - array([ 0., 0.04021286, 0.50978714, 1., 0.50978714, 0.04021286, 0., 0.])
    assert absolute(d).max() < 1e-8


#def test_PxC(N=50):
#    for n in range(N):
#        r = 5*(2*random(10)-1)
#        l = 10*random()
#        w = _weight_blackman(r,l)
#        cw = c_weight_blackman(r,l)
#        assert type(w) == type(cw)
#        assert absolute(w - cw).max() < 1e-10


#def test_cython_scalar():
#    R = array([-3, -2, -1, 0, 1, 2, 3, 100])
#    W = array([ 0., 0.0954915, 0.6545085, 1., 0.6545085, 0.0954915, 0., 0.])
#    l = 5
#    for r, w in zip(R, W):
#        w2 = _weight_blackman_scalar(r, l)
#        d = w - w2
#        assert absolute(d) < 1e-8

def out_of_window():
    r = 5*(2*random(10)-1)
    l = 10*random()
    w = _weight_blackman(r,l)
#    cw = c_weight_blackman(r,l)
    ind = r>l/2
    assert (w[ind]==0).all()
    assert (cw[ind]==0).all()

# Question: _weight_blackman(ma.masked_all(3), 5) should return a masked array?
# _weight_blackman(ma.masked_all(4),5)
