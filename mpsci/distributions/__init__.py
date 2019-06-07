"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

__all__ = ['beta', 'cosine', 'exponweib', 'gamma', 'genextreme', 'geninvgauss',
           'lognormal', 'ncx2', 'normal', 'rice',
           'hypergeometric', 'fishers_noncentral_hypergeometric']

from .continuous import *
from .discrete import *
