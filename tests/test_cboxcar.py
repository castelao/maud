
def test_answer():
    from maud.cwindow_func import _weight_boxcar
    from numpy import array, absolute
    r = array([-3, -2.1, -1, 0, 1, 2.6, 3, 100])
    l = 5
    w = _weight_boxcar(r,l)
    a = array([ 0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    assert (w == a).all()

