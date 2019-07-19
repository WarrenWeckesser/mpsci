"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from . import (beta, cosine, exponweib, f, fishers_noncentral_hypergeometric,
               gamma, genextreme, geninvgauss, hypergeometric, lognormal,
               multivariate_hypergeometric, ncx2, negative_binomial, normal,
               rice)
