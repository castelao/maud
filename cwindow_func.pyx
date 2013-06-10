# http://docs.cython.org/src/userguide/numpy_tutorial.html#numpy-tutorial
import numpy as np
#cimport numpy as np
from numpy import pi

# Hamming
def _weight_hamming(r, double l):
    """
    """
    cdef int N, n
    cdef double pi
    N = r.shape[0]
    w = np.zeros(N, dtype=r.dtype)
    for n in range(N):
        if np.absolute(r[n])<=l:
            w[n]=0.54+0.46*np.cos(pi*r[n]/l)
    return w
