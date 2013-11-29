
def test_answer():
    from maud.window_func import _weight_hann
    from numpy import array, absolute
    r = array([-3, -2, -1, 0, 1, 2, 3, 100])
    l = 5
    w = _weight_hann(r,l)
    d = w - array([ 0., 0.0954915, 0.6545085, 1., 0.6545085, 0.0954915, 0., 0.])
    assert absolute(d).max()<1e-8

