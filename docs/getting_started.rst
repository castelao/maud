*************************
Getting Started with MAUD
*************************

One can use MAUD inside Python or in the shell.

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

Shell script
------------

To check the available options

    >>> maud4latlonnc -h

On this example, it will filter the variable 'temperature' at the netCDF file model_output.nc. The weight function used is hamming, and the window scale is 600e3. The npes define the number of parallel process to be used, in this case 18. The option --iterp defines that any masked value will be replaced in the output as the filtered result of the valid values around it, inside the window lenght.

    >>> maud4latlonnc -l 600e3 --var='temperature' -w hamming --interp --npes=18 model_output.nc
