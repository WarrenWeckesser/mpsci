"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from . import (benktander1, benktander2, beta, binomial, cosine, exponweib, f,
               fishers_noncentral_hypergeometric, gamma, gamma_gompertz,
               genexpon, genextreme, geninvgauss, genpareto, hypergeometric,
               laplace, levy, logistic, lognormal, multivariate_hypergeometric,
               ncf, ncx2, negative_binomial, negative_hypergeometric, normal,
               poisson, rice, t)
