*************************
Getting Started with MAUD
*************************

One can use MAUD inside Python or in the shell.

Shell script
============

MAUD provides two shell commands, maud4nc for 1D filter and maud4latlonnc for 2D filter on geographic coordinates.

maud4nc
-------

To check the available options::

    >>> maud4nc -h

In the example below, the variable temperature (--var), at the netCDF file model_output.nc, is filtered along the time (--scalevar) using a hann window (-w), and the output will be saved at model_highpass.nc (-o). This is a bandpass filter (--highpasswindowlength together with --lowpasswindowlength), preserving scales between 120 and 10 units of the scalevar (on this case: time).::

    >>> maud4nc --highpasswindowlength=120 --lowpasswindowlength=10 --scalevar=time \
    >>> --var='temperature' -w hann -o model_highpass.nc model_output.nc

maud4latlonnc
-------------

To check the available options::

    >>> maud4latlonnc -h

In the example below, the variable temperature (--var), at the netCDF file model_output.nc, is filtered along the space (lat x lon). The variables latitude and longitude must exist in the same file. This is a lowpass filter (--largerpasslength), hence it attenuates eveything with spatial scale smaller than 600e3 meters. The weights are defined by a hamming function (-w). The npes define the number of parallel process to be used, in this case 18. The option --interp defines that any missing value will be replaced in the output as the filtered result of the valid values around it, inside the window lenght.::

    >>> maud4latlonnc --largerpasslength=600e3 --var='temperature' \
    >>> -w hamming --interp --npes=18 -o model_highpass model_output.nc

Inside Python
=============

    >>> from maud import window_1Dmean, window_mean_2D_latlon
    >>> window_1Dmean(x, l=200e3, t=None, method='hann', axis=0, parallel=True)

    >>> window_mean_2D_latlon(Lat, Lon, data, l, method='hamming', interp=False)

The faster version
------------------

There is a Cython version of each filter. If you're able to, use cmaud instead of maud to gain at least one order of magnitude on the speed.

    >>> from cmaud import window_1Dmean
    >>> window_1Dmean(x, l=200e3)

    >>> from cmaud import window_mean_2D_latlon
    >>> window_mean_2D_latlon(Lat, Lon, data, l)

