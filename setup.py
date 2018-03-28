import os
import sys
import imp

from setuptools import setup, find_packages

try:
    __doc__ = open('README.md').read()
except IOError:
    pass

__file__ = './'
ROOT            = 'binf'
LOCATION        = os.path.abspath(os.path.dirname(__file__))
JUNK            = ['CVS']

NAME            = "binf"
VERSION         = "0.1"
AUTHOR          = "Simeon Carstens"
EMAIL           = "simeon.carstens@mpibpc.mpg.de"
URL             = "http://www.simeon-carstens.de"
SUMMARY         = "Some classes faciliating Bayesian inference tasks occuring in ISD"
DESCRIPTION     = __doc__
LICENSE         = 'MIT'
REQUIRES        = ['numpy', 'csb']

setup(
    name=NAME,
    packages=find_packages(exclude=('tests',)),
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=SUMMARY,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    classifiers=(
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries')
    )

