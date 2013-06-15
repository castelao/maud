# http://docs.cython.org/src/userguide/numpy_tutorial.html#numpy-tutorial

import numpy as np
from numpy import ma

cimport numpy as np
from libc.math cimport cos

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# Hamming
def _weight_hamming(np.ndarray r, double l):
    """ Cython hamming weight

        Input:
	    r np.array
	    l double

	w = 0.54 - 0.46*cos(2*pi*n/(N-1))
	where n is the element index of a total N elements.
	or
	w = 0.54 + 0.46*cos(2*pi*r/l),
	where r is the distance to the center of the window,
	and l is the total width of the filter window.
	hint: cos(a-b) = cos(a)cos(b)+sin(a)sin(b)

    """
    assert r.dtype == DTYPE
    cdef int n
    cdef int N = len(r)
    cdef double lhalf = l/2.
    cdef double scale = 2*np.pi/l
    cdef np.ndarray w = np.zeros(N, dtype=DTYPE)
    for n in range(N):
        if abs(r[n])<=lhalf:
            w[n] = 0.54 + 0.46*cos(scale*r[n])
    return w


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
