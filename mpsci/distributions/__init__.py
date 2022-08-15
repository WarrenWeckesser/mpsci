"""
``distributions``
-----------------

A few probability distributions.

Note that these implementations do not necessarily use the same parametrization
as the corresponding implementations in `scipy.stats`.

"""

from . import (argus, benktander1, benktander2, beta, binomial, burr12, chi,
               chi2, cosine, exponweib, f, fishers_noncentral_hypergeometric,
               gamma, gamma_gompertz, gauss_kuzmin, genexpon, genextreme,
               geninvgauss, genpareto, gumbel_max, gumbel_min, hypergeometric,
               invchi2, invgauss, kumaraswamy, laplace, levy, loggamma,
               logistic, loglogistic, lognormal, logseries,
               multivariate_hypergeometric, multivariate_t, nakagami, ncf, nct,
               ncx2, negative_binomial, negative_hypergeometric, normal,
               poisson, rice, slash, t, truncnorm, uniform, vonmises,
               weibull_max, weibull_min)
