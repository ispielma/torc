from setuptools import setup
import os

__version__ = '0.1.1'

DESCRIPTION = """Field and gradient calculations for magnetic coils"""

# Auto generate a __version__ package for the package to import
with open(os.path.join('torc', '__version__.py'), 'w') as f:
    f.write("__version__ = '%s'\n" % __version__)

setup(
    name='torc',
    version=__version__,
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chris Billington',
    author_email='chrisjbillington@gmail.com',
    url='https://bitbucket.org/cbillington/torc',
    license="BSD",
    packages=["torc"],
    install_requires=["numpy", "scipy"],
)
