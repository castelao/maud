
def test_answer():
    from maud.window_func import _weight_boxcar
    from cmaud.window_func import _weight_boxcar as _cweight_boxcar 
    from numpy import array, absolute
    r = array([-3, -2.1, -1, 0, 1, 2.6, 3, 100])
    l = 5
    w = _weight_boxcar(r,l)
    wc = _cweight_boxcar(r,l)
    a = array([ 0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    assert (w == a).all()
    assert (wc == a).all()

