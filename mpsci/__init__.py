"""
``mpsci`` is a Python package that defines an assortment of numerical
formulas and algorithms.  The library ``mpmath`` is used for floating point
calculations.

"""

import importlib.metadata
from . import distributions
from . import fun
from . import polyapprox
from . import signal
from . import stats


__version__ = importlib.metadata.version("mpsci")
