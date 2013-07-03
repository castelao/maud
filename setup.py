# -*- coding: utf-8 -*-
try:
    from setuptools import setup, find_packages
except ImportError:
    import distribute_setup
    distribute_setup.use_setuptools()
    from setuptools import setup, find_packages

import os
#import sys
from distutils import log

#from distutils.core import setup
# Which Extension to use?
#from distutils.extension import Extension
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()

version='0.6.3'

requires = [
    'numpy>=1.1',
    'fluid>=0.2',
    ]

setup(
    name='maud',
    version=version,
    url='http://maud.castelao.net',
    download_url='http://pypi.python.org/packages/source/m/maud/maud-'+version+'.tar.gz',
    license='PSF',
    author='Guilherme Castelao, Bia Villas-Boas, Luiz Irber',
    author_email='guilherme@castelao.net, bia@melovillasboas.com, luiz.irber@gmail.com',
    description='Moving Average for Uneven Data',
    long_description=README + '\n\n' + NEWS,
    #zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Python Software Foundation License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    platforms='any',
    #py_modules=['maud', 'window_func'],
    packages=find_packages(),
    #packages=['maud', 'maud.window_func'],
    install_requires=requires,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("maud.cwindow_func", ["maud/window_func.pyx"]), Extension("cmaud", ["maud/maud.pyx"])],
    include_dirs = [np.get_include()],
    #ext_modules = [
    #    Extension("maud.cwindow_func", ["window_func.pyx"],
    #    #libraries=['maud'],
    #    include_dirs = [np.get_include()],
    #    #pyrex_include_dirs=['.']
    #    ),
    #    ],
    scripts=["bin/maud4nc"],
)

