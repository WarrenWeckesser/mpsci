"""
Fisher's noncentral hypergeometric distribution
-----------------------------------------------

*Preliminary version.  The parameters may change.*

"""

import mpmath
from . import hypergeometric


__all__ = ['support', 'mode', 'mean']


def support(ncp, ntotal, ngood, nsample):
    """
    *Full* PMF for Fisher's noncentral hypergeometric distribution.

    Requires 0 < ncp < inf.

    Returns
    -------
    sup : range
        The support of the distribution, represented as a Python range.
    pmf : list
        The values of the probability mass function on the support.

    Examples
    --------
    >>> import mpmath
    >>> from mpsci.distributions import fishers_noncentral_hypergeometric
    >>> mpmath.mp.dps = 24
    >>> sup, pmf = fishers_noncentral_hypergeometric(2.5, 16, 8, 10)
    >>> sup
    range(2, 9)
    >>> pmf
    [mpf('0.000147721056482530831000186887'),
     mpf('0.00590884225930123324000747648'),
     mpf('0.0646279622111072385625817892'),
     mpf('0.258511848844428954250327208'),
     mpf('0.403924763819420241016136111'),
     mpf('0.230814150753954423437792171'),
     mpf('0.0360647110553053786621550275')]

    """
    support, values = hypergeometric.support(ntotal, ngood, nsample)
    lpmf = [hypergeometric.logpmf(k, ntotal, ngood, nsample) for k in support]

    g = [lpmf[k - support[0]] + mpmath.log(ncp) * k for k in support]
    gmax = max(g)
    g = [mpmath.exp(v - gmax) for v in g]
    gsum = mpmath.fsum(g)
    values = [v/gsum for v in g]
    return support, values


support._docstring_re_subs = [
    (' inf[.]', r':math:`\\infty`.', 0, 0),
]


def mode(ncp, ntotal, ngood, nsample):
    """
    Mode of Fisher's noncentral hypergeometric distribution.

    Returns
    -------
    m : mpmath.mpf
        The mode of the distribution.

    Examples
    --------
    >>> from mpsci.distributions import fishers_noncentral_hypergeometric
    >>> fishers_noncentral_hypergeometric.mode(2.5, 16, 8, 10)
    mpf('6.0')

    """
    ncp = mpmath.mpf(ncp)
    A = ncp - 1
    B = ngood + nsample - ntotal - (ngood + nsample + 2)*ncp
    C = (ngood + 1)*(nsample + 1)*ncp
    m = mpmath.floor(-2*C / (B - mpmath.sqrt(B**2 - 4*A*C)))
    return m


def mean(ncp, ntotal, ngood, nsample):
    """
    Mode of Fisher's noncentral hypergeometric distribution.

    This calculation is implemented as a weighted sum over the support.
    It may be very slow for large parameters.

    Returns
    -------
    m : mpmath.mpf
        The mean of the distribution.

    Examples
    --------
    >>> import mpmath
    >>> from mpsci.distributions import fishers_noncentral_hypergeometric
    >>> mpmath.mp.dps = 24
    >>> fishers_noncentral_hypergeometric.mean(2.5, 16, 8, 10)
    mpf('5.89685838859408792634258808')
    """
    sup, p = support(ncp, ntotal, ngood, nsample)
    n = len(p)
    with mpmath.extradps(5):
        return mpmath.fsum([sup[k]*p[k] for k in range(len(p))])
