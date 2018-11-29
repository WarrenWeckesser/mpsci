
import setuptools
from setuptools import setup
from os import path


def get_version():
    """
    Find the value assigned to __version__ in mpsci/__init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in that file.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("mpsci/__init__.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]

# Get the long description from README.rst.
_here = path.abspath(path.dirname(__file__))
with open(path.join(_here, 'README.rst')) as f:
    _long_description = f.read()


setup(
    name='mpsci',
    version=get_version(),
    author='Warren Weckesser',
    description=("Assorted statistical and signal processing functions "
                 "implemented using mpmath."),
    long_description=_long_description,
    url="https://github.com/WarrenWeckesser/mpsci",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        'mpmath',
    ],
    keywords="mpmath scipy",
)
