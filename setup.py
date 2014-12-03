# -*- coding: utf-8 -*-

NAME = 'maud'
VERSION = '0.9.3'
DESCRIPTION = 'Moving Average for Uneven Data'
#here = os.path.abspath(os.path.dirname(__file__))
#README = open(os.path.join(here, 'README.rst')).read()
README = ""
#NEWS = open(os.path.join(here, 'NEWS.txt')).read()
NEWS = ""
LONG_DESCRIPTION = README + '\n\n' + NEWS
AUTHOR = 'Guilherme Castelao, Bia Villas-Boas, Luiz Irber',
AUTHOR_EMAIL = 'guilherme@castelao.net, bia@melovillasboas.com, luiz.irber@gmail.com',
LICENSE = 'PSFL'
PLATFORMS = 'any'
URL = 'http://maud.castelao.net'
DOWNLOAD_URL = 'http://pypi.python.org/packages/source/m/maud/maud-%s.tar.gz' % (VERSION),
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Python Software Foundation License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

import sys, os.path

from distutils import log
from distutils.core import setup#, Command
#from distutils.core import Distribution as _Distribution
from distutils.core import Extension as _Extension
#from distutils.dir_util import mkpath
from distutils.command.build_ext import build_ext as _build_ext
#from distutils.command.bdist_rpm import bdist_rpm as _bdist_rpm
#from distutils.errors import CompileError, LinkError, DistutilsPlatformError

if 'setuptools.extension' in sys.modules:
    _Extension = sys.modules['setuptools.extension']._Extension
    sys.modules['distutils.core'].Extension = _Extension
    sys.modules['distutils.extension'].Extension = _Extension
    sys.modules['distutils.command.build_ext'].Extension = _Extension


#if sys.version_info[0] < 3:
#    try:
#        from Cython.Distutils.extension import Extension as _Extension
#        from Cython.Distutils import build_ext as _build_ext
#    except:
#        pass


try:
    from setuptools import setup, find_packages
except ImportError:
    import distribute_setup
    distribute_setup.use_setuptools()
    from setuptools import setup, find_packages


#from distutils.core import setup
# Which Extension to use?
#from distutils.extension import Extension
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

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

requires = [
    'numpy>=1.1',
    'distribute>=0.6.40',
    ]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README + '\n\n' + NEWS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    platforms=PLATFORMS,
    url=URL,
    download_url=DOWNLOAD_URL,
    classifiers=CLASSIFIERS,
    zip_safe=False,
    packages=find_packages(),
    install_requires=requires,
    cmdclass = {'build_ext': build_ext, 'test': PyTest},
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
    scripts=["bin/maud4nc", "bin/maud4latlonnc"],
    tests_require=['pytest'],
)

