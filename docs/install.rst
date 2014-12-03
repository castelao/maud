************
Installation
************

Requirements
============

- `Python <http://www.python.org/>`_ 2.6 (>=2.6.5), 2.7, 3.1 or 3.2

- `Numpy <http://www.numpy.org>`_

- distribute 0.6 (>=0.6.40)

- Cython (>= 0.20)

Installing MAUD
==================

Using pip
---------

To install the dependencies::

    $ pip install numpy
    $ pip install distribute
    $ pip install cython

To install MAUD itself::

    $ pip install --no-deps maud

.. note::

    The ``--no-deps`` flag is optional, but highly recommended if you already
    have Numpy installed, since otherwise pip will sometimes try to "help" you
    by upgrading your Numpy installation, which may not always be desired.
