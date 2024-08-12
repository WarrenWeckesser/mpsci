"""
``distributions``
-----------------

An assortment of probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from . import (argus, benktander1, benktander2, beta, betabinomial, betaprime,
               binomial, burr12, cauchy, chi, chi2, cosine, dirichlet,
               exponweib, f,
               fishers_noncentral_hypergeometric, folded_normal,
               gamma, gamma_gompertz,
               gauss_kuzmin, genexpon, genextreme, genhyperbolic, geninvgauss,
               genpareto, gompertz, gumbel_max, gumbel_min,
               half_logistic, hypergeometric, hypsecant, invchi2, invgamma,
               invgauss, kumaraswamy, laplace, levy,
               loggamma, logistic, loglogistic, lognormal, logseries,
               maxwell, multivariate_hypergeometric, multivariate_t,
               nakagami, ncf, nct, ncx2, negative_binomial,
               negative_hypergeometric, normal, pareto, poisson, power_normal,
               rel_breitwigner, rice, slash, studentt, truncnorm, uniform,
               vonmises, weibull_max, weibull_min)
from ._common import Initial
