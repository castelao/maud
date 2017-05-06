#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE


import sys, os.path

from distutils import log
from distutils.core import setup#, Command
#from distutils.core import Distribution as _Distribution
from distutils.core import Extension# as _Extension
#from distutils.dir_util import mkpath
from distutils.command.build_ext import build_ext #as _build_ext
#from distutils.command.bdist_rpm import bdist_rpm as _bdist_rpm
#from distutils.errors import CompileError, LinkError, DistutilsPlatformError

if 'setuptools.extension' in sys.modules:
    _Extension = sys.modules['setuptools.extension']._Extension
    sys.modules['distutils.core'].Extension = _Extension
    sys.modules['distutils.extension'].Extension = _Extension
    sys.modules['distutils.command.build_ext'].Extension = _Extension


try:
    #from setuptools import setup, find_packages
    from setuptools import find_packages
except ImportError:
    import distribute_setup
    distribute_setup.use_setuptools()
    #from setuptools import setup, find_packages
    from setuptools import find_packages


with_cython = None
if sys.version_info[0] < 3:
    try:
        from Cython.Distutils.extension import Extension# as _Extension
        from Cython.Distutils import build_ext# as _build_ext
        import numpy as np
        with_cython = True
    except ImportError:
        import time
        print("Couldn't find Cython. We strongly recommend to install it to "
                "increase MAUD's performance")
        time.sleep(5)

# ============================================================================
from setuptools.command.test import test as TestCommand
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

# ============================================================================
setup_args = {
    'name': 'maud',
    'description': 'Moving Average for Uneven Data',
    'author': 'Guilherme Castelao, Bia Villas-Boas, Luiz Irber',
    'author_email': 'guilherme@castelao.net, bia@melovillasboas.com, luiz.irber@gmail.com',
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'
        ],
    'cmdclass': {
        'build_ext': build_ext,
        'test': PyTest
        },
    'keywords': 'filter, uneven data, gaps',
    'license': 'BSD license',
    'scripts': ['bin/maud4nc', 'bin/maud4latlonnc'],
    'tests_require': ['pytest'],
    'url': 'http://maud.castelao.net',
    'zip_safe': False,
    }

with open('VERSION') as version_file:
    setup_args['version'] = version_file.read().rstrip('\n')

with open('README.rst') as readme_file:
    readme = readme_file.read()
with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')
setup_args['long_description'] = readme + '\n\n' + history

with open('requirements.txt') as requirements_file:
    setup_args['install_requires'] = requirements_file.read()

if __name__ == '__main__':

    if with_cython:
        setup(
            packages=find_packages(),
            ext_modules = [
                Extension("cmaud", ["maud/maud.pyx"]),
                Extension("maud.cwindow_func", ["maud/window_func.pyx"]),
                Extension("maud.cwindow_func_scalar", ["maud/window_func_scalar.pyx"]),
                Extension("maud.cdistance", ["maud/distance.pyx"]),
                ],
            include_dirs = [np.get_include()],
            #ext_modules = [
            #    Extension("maud.cwindow_func", ["window_func.pyx"],
            #    #libraries=['maud'],
            #    include_dirs = [np.get_include()],
            #    #pyrex_include_dirs=['.']
            #    ),
            #    ],
            **setup_args
        )
    else:
        setup(
            packages=[
                'maud',
                ],
            package_dir={
                'maud': 'maud'},
            **setup_args
        )
