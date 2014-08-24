""" Weight functions"""

import numpy as np
from numpy import ma
from numpy import pi

# defining some weight functions

def window_func(method='hamming'):
    """ Select the weight function
    """
    assert type(method) == str, 'method must be a string'

    if method == 'hamming':
        return _weight_hamming

    elif method == 'hann':
        return _weight_hann

    elif method == 'blackman':
        return _weight_blackman

    elif method == 'triangular':
        return _weight_triangular

    elif method == 'boxcar':
        return _weight_boxcar

    raise Exception('%s is not available' % method)

# Hamming
def _weight_hamming(r, l):
    """ Hamming weight

        w = 0.54 - 0.46*cos(2*pi*n/(N-1))
        where n is the element index of a total N elements.
        or
        w = 0.54 + 0.46*cos(2*pi*r/l),
        where r is the distance to the center of the window,
        and l is the total width of the filter window.
        hint: cos(a-b) = cos(a)cos(b)+sin(a)sin(b)
    """
    w = 0.54 + 0.46*np.cos(2*pi*r/l)
    w[np.absolute(r)>l/2.]=0
    return w


# hann
def _weight_hann(r,l):
    """ Hann weight

        from the definition of the Hann window centered in n = (N-1)/2:
        w = 0.5*(1 + np.cos(2*pi*n/(N-1))),
        where n is the element index of a toal of N elements.
        To make it symmetrical, i.e centered in n=0, we should add a
        phase of pi in the cosine argument. So,
        w = 0.5*(1 - np.cos(2*pi*r/l)),
        where r is the distance to the center of teh window
        and l is the total width of the window.
    """
    w = 0.5 * (1 + np.cos(2*pi*r/l))
    w[np.absolute(r) > l/2.] = 0
    return w


# blackman
def _weight_blackman(r,l):
    """ 
    using values from wikipedia 'exact blackman'
    
    """
    w = 0.42 +  0.5*np.cos(2*pi*r/l) + 0.08*np.cos(4*pi*r/l) # fase lag-> sign change
    w[np.absolute(r)>l/2.]=0
    return w


# triangular
def _weight_triangular(r,l):
    """
    """
    w = ma.masked_all(r.shape)
    ind = np.absolute(r)<l/2.
    ind2 = np.absolute(r)>l/2.
    w[ind] = 1-np.absolute(2*r[ind]/l)
    w[ind2] = 0
    return w


# boxcar or rectangular
def _weight_boxcar(r, l):
    """
    """
    w = np.zeros(r.shape)
    w[np.abs(r)<=l/2.] = 1
    return w


# 2D functions

# hamming 2D
def _weight_hamming_2D(x, y, l):
    """
        check _weight_hamming
    """
    r = (x**2+y**2)**0.5
    w = 0.54 + 0.46*np.cos(2*pi*r/l)
    w[np.absolute(r)>l/2.] = 0
    return w

# hann 2D
def _weight_hann_2D(x,y,l):
    """
        Check _weight_hann 
    """
    r=(x**2+y**2)**0.5
    w=0.5*(1+np.cos(2*pi*r/l))
    w[r>l]=0
    return w


# hann 1D bandpass
def _weight_hann_band(r,l1,l2):
    """ Hann weight 
        
        from the definition of the Hann window centered in n = (N-1)/2:
        w = 0.5*(1 + np.cos(2*pi*n/(N-1))),
        where n is the element index of a toal of N elements.
        To make it symmetrical, i.e centered in n=0, we should add a 
        phase of pi in the cosine argument. So,
        w = 0.5*(1 - np.cos(2*pi*r/l)),
        where r is the distance to the center of teh window
        and l1 is the width of the first low-pass window and l2
        is the width of the second low-pass window.
        So, the band-pass is going to be for l2 > f > l1.
    """
    w= ((np.cos(pi*r/l1)**2))*((np.sin(pi*r/l2))**2)
    w[np.absolute(r)>l1/2.]=0
    w[np.absolute(r)<l2/2.]=0
    return w


# lanczos 2D
def _weight_lanczos_2D(x,y,l,cutoff):
    """ Working on
    """
    #c=cutoff
    r=(x**2+y**2)**0.5
    w=np.sinc(r/cutoff)*np.sinc(r/cutoff/l)
    w[r>3*l]=0

# ==== funcoes antigas ======================

def _weight_triangular_2D(x,y,l):
    """
    """
    r=(x**2+y**2)**0.5
    w=(l-r)/l
    w[r>l]=0
    return w
