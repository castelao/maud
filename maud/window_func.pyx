# -*- coding: utf-8 -*-
# cython: profile=False
# http://docs.cython.org/src/userguide/numpy_tutorial.html#numpy-tutorial

import numpy as np
from numpy import ma

cimport numpy as np
from libc.math cimport cos, abs, M_PI

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double pi = np.pi
cdef double twopi = 2*M_PI



def window_func(method='hamming'):
    """ Select the weight function
    """
    if method == 'hamming':
        winfunc = _weight_hamming

    elif method == 'hann':
        winfunc = _weight_hann

    #elif method == 'blackman':
    #    winfunc = _weight_blackman

    else:
        raise Exception("%s is not available" % method)

    #elif method == 'triangular':
    #    winfunc = _weight_triangular

    return winfunc


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
    #print r.dtype
    #print DTYPE
    #assert r.dtype == DTYPE
    cdef int n
    cdef int N = len(r)
    cdef double lhalf = l/2.
    cdef double scale = 2*np.pi/l
    cdef np.ndarray w = np.zeros(N, dtype=DTYPE)
    for n in range(N):
        if abs(r[n])<=lhalf:
            w[n] = 0.54 + 0.46*cos(scale*r[n])
    return w

def _weight_hann(np.ndarray r, double l):
    """ Cython Hann weight

        Input:
            r np.array
            l double

	from the definition of the Hann window centered in n = (N-1)/2:
        w = 0.5*(1 - np.cos(2*pi*n/(N-1))),
        where n is the element index of a toal of N elements.
        To make it symmetrical, i.e centered in n=0, we should add a 
        phase of pi in the cosine argument. So,
        w = 0.5*(1 + np.cos(2*pi*r/l)),
        where r is the distance to the center of teh window
        and l is the total width of the window.

    """
    #print r.dtype
    #print DTYPE
    #assert r.dtype == DTYPE
    cdef int n
    cdef int N = len(r)
    cdef double lhalf = l/2.
    cdef double scale = 2*np.pi/l
    cdef np.ndarray w = np.zeros(N, dtype=DTYPE)
    for n in range(N):
        if abs(r[n])<=lhalf:
            w[n] = 0.5*(1 + cos(scale*r[n]))
    return w
