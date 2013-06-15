# http://docs.cython.org/src/userguide/numpy_tutorial.html#numpy-tutorial

import numpy as np
from numpy import ma
cimport numpy as np
from numpy import pi


def window_func(method='hamming'):
    """ Select the weight function
    """
    if method == 'hamming':
        winfunc = _weight_hamming

    #elif method == 'hann':
    #    winfunc = _weight_hann

    #elif method == 'blackman':
    #    winfunc = _weight_blackman

    #elif method == 'triangular':
    #    winfunc = _weight_triangular

    return winfunc


# Hamming
def _weight_hamming(r, double l):
    """
    """
    #print "c hamming"
    cdef int N, n
    cdef double pi
    N = r.shape[0]
    w = ma.zeros(N, dtype=r.dtype)
    for n in range(N):
        #if np.absolute(r[n])<=l:
        if (r[n]>=l) & (r[n]<=l):
            w[n]=0.54+0.46*np.cos(pi*r[n]/l)
    return w
