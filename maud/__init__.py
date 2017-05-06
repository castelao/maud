# -*- coding: utf-8 -*-

__author__ = 'Gui Castelão'
__email__ = 'guilherme@castelao.net'
__version__ = '0.10.0'

__all__ = ['core', 'window_func', 'distance']

from .core1D import wmean_1D_serial, wmean_1D, wmean_bandpass_1D_serial, wmean_bandpass_1D
from .core2D import wmean_2D_serial, wmean_2D
from .core2Dlatlon import wmean_2D_latlon_serial, wmean_2D_latlon
