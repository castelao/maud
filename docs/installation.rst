.. highlight:: shell

============
Installation
============

Requirements
------------

- `Python <http://www.python.org/>`_ 2.6 (>=2.6.5), 2.7, 3.1 or 3.2

- `Numpy <http://www.numpy.org>`_

Strongly recommended:

- distribute 0.6 (>=0.6.40)

- Cython (>= 0.20)


Stable release
--------------

Optional: MAUD can run without Cython, but it's strongly recommended since some filters can run more than 10 times faster with Cython. In that case it must be installed before install MAUD itself. To do so, run this command in your terminal:

.. code-block:: console

    $ pip install distribute>=0.6.40
    $ pip install cython

To install MAUD, run this command in your terminal:

.. code-block:: console

    $ pip install maud

This is the preferred method to install MAUD, as it will always install the most recent stable release. 
Cython is not required, but it's strongly recommend since some of MAUD filters can run more than 10 times faster.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for MAUD can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/castelao/maud

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/castelao/maud/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/castelao/maud
.. _tarball: https://github.com/castelao/maud/tarball/master
