# -*- coding: utf-8 -*-
# cython: profile=False
# filename: window_func_scalar.pyx
# http://docs.cython.org/src/userguide/numpy_tutorial.html#numpy-tutorial

import numpy as np

cimport numpy as np
from libc.math cimport cos, abs, M_PI


cdef double twopi = 2*M_PI


def window_func_scalar(method='hamming'):
    """ Select the weight function
    """
    if method == 'hamming':
        return _weight_hamming_scalar

    elif method == 'hann':
        return _weight_hann_scalar

    elif method == 'blackman':
        return _weight_blackman_scalar

    elif method == 'boxcar':
        return _weight_boxcar_scalar

    else:
        raise Exception("%s is not available" % method)

    #elif method == 'triangular':
    #    winfunc = _weight_triangular_scalar


# Hamming
def _weight_hamming_scalar(double r, double l):

    if abs(r) > l/2.:
        return 0

    return 0.54 + 0.46*cos(twopi*r/l)


# Hann
def _weight_hann_scalar(double r, double l):
    """ Need to double check and create a test
    """
    if abs(r) > l/2.:
        return 0

    return 0.5*(1 + cos(twopi*r/l))


# blackman
def _weight_blackman_scalar(double r, double l):
    """ Need to double check and create a test
    """
    if abs(r) > l/2.:
        return 0

    cdef double theta = twopi*r/l
    # fase lag-> sign change
    return 0.42 +  0.5*cos(theta) + 0.08*cos(2*theta)


# triangular
#def _weight_triangular_scalar(double r, double l):
#    """ Need to double check and create a test
#    """
#    if abs(r) > l/2.:
#        return 0
#
#    return 1 - abs(2*r/l)


# boxcar or rectangular
def _weight_boxcar_scalar(double r, double l):
    """ Need to double check and create a test
    """
    if abs(r) > l/2.:
        return 0

    return 1
