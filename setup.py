#!/usr/bin/env python

# To upload a version to PyPI, run:
#    python setup.py sdist upload
# If the package is not registered with PyPI yet, do so with:
#     python setup.py register

import sys
if 'sdist' in sys.argv:
    import pypandoc
    with open('README', 'w') as f:
        f.write(pypandoc.convert('README.md', 'rst', format='markdown'))

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#TODO: dependencies

__version__ = '0.1.0'

DESCRIPTION = """Field and gradient calculations for magnetic coils"""

# Auto generate a __version__ package for the package to import
with open(os.path.join('torc', '__version__.py'), 'w') as f:
    f.write("__version__ = '%s'\n" % __version__)

setup(name='torc',
      version=__version__,
      description=DESCRIPTION,
      # long_description=open('README').read(),
      author='Chris Billington',
      author_email='chrisjbillington@gmail.com',
      url='https://bitbucket.org/cbillington/torc',
      license="BSD",
      packages=["torc"]
      )
