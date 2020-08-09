"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from . import (beta, binomial, cosine, exponweib, f,
               fishers_noncentral_hypergeometric, gamma, gamma_gompertz,
               genexpon, genextreme, geninvgauss, genpareto, hypergeometric,
               laplace, logistic, lognormal, multivariate_hypergeometric, ncx2,
               negative_binomial, negative_hypergeometric, normal, poisson,
               rice, t)
