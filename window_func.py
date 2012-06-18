""" Weight functions"""

import numpy as np
from numpy import ma
from numpy import pi

from fluid.common.distance import distance

# defining some weight functions

# blackman
def _weight_blackman(r,l):
    """ 
    using values from wikipedia 'exact blackman'
    
    """
    w = 0.42 +  0.5*np.cos(2*pi*r/l) + 0.08*np.cos(4*pi*r/l) # fase lag-> sign change
    w[np.absolute(r)>l]=0
    return w

# triangular
def _weight_triangular_2D(x,y,l):
    """
    """
    r=(x**2+y**2)**0.5
    w=(l-r)/l
    w[r>l]=0
    return w

# hamming
def _weight_hamming(r,l):
    """
    """
    w=0.54+0.46*numpy.cos(pi*r/l)
    w[numpy.absolute(r)>l]=0
    return w

# hamming 2D
def _weight_hamming_2D(x, y, l):
    """
    """
    r = (x**2+y**2)**0.5
    w = 0.54 + 0.46*np.cos(2*pi*r/l)
    w[r>l] = 0
    return w

# hann
def _weight_hann(r,l):
    """
    """
    w=0.5*(1+np.cos(pi*r/l))
    w[np.absolute(r)>l]=0
    return w

# hann 2D
def _weight_hann_2D(x,y,l):
    """
    """
    r=(x**2+y**2)**0.5
    w=0.5*(1+np.cos(pi*r/l))
    w[r>l]=0
    return w

# lanczos 2D
def _weight_lanczos_2D(x,y,l,cutoff):
    """ Working on
    """
    #c=cutoff
    r=(x**2+y**2)**0.5
    w=np.sinc(r/cutoff)*np.sinc(r/cutoff/l)
    w[r>3*l]=0
