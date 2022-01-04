"""
``fun``
-------

This module contains an assortment of functions that arise
in scientific computing.
"""

from ._boxcox import boxcox, boxcox1p
from ._yeo_johnson import yeo_johnson, inv_yeo_johnson
from ._marcumq import marcumq, cmarcumq
from ._digammainv import digammainv
from ._xlogy import xlogy, xlog1py
from ._logbeta import logbeta
from ._logbinomial import logbomial
from ._logsumexp import logsumexp
from ._legendre import roots_legendre
