"""
Fisher's noncentral hypergeometric distribution
-----------------------------------------------

*Preliminary version.  The parameters may change.*

"""

from functools import lru_cache
import re
from mpmath import mp
from . import hypergeometric as _hg


__all__ = ['support_pmf', 'pmf_dict', 'pmf', 'cdf', 'sf', 'mode', 'mean']


# This auxiliary version of support includes the parameter prec,
# so the mpmath precision is part of the cache key.
@lru_cache()
def _support_pmf(nc, ntotal, ngood, nsample, prec):
    with mp.extradps(5):
        support = _hg.support(ntotal, ngood, nsample)
        lpmf = [_hg.logpmf(k, ntotal, ngood, nsample)
                for k in support]

        # The PMF of Fisher's noncentral hypergeometric distribution is
        # proportional to a weighted version of the hypergeometric
        # distribution.  The weights are the powers of the noncentrality
        # parameter.  To maintain precision over a wide range of values, we
        # compute the log of the weighted hypergeometric PMF:
        g = [lpmf[k - support[0]] + mp.log(nc) * k for k in support]

        # g contains the logs of values proportional to the noncentral
        # hypergeometric PMF.  That is, g = [log(c0), log(c1), log(c2), ...].
        # We must exponentiate these values, and then normalize them to get
        # a PMF.  To exponentiate safely, we'll subtract the maximum value
        # from all the values before exponentiating.  So instead of computing
        # [exp(log(c0), exp(log(c1)), exp(log(c2)), ...], we compute
        #   [exp(log(c0) - log(cmax)),
        #    exp(log(c1) - log(cmax)),
        #    exp(log(c2) - log(cmax)),
        #    ...],
        # which is
        #   [c0/cmax, c1/cmax, c2/cmax, ...].
        # and the maximum value in that sequence is therefore 1.
        gmax = max(g)
        eg = [mp.exp(v - gmax) for v in g]

        # The values in eg are proportional to the desired PMF, and the
        # maximum value in eg is 1.  Divide all the values in eg by sum(eg)
        # to create a PMF.
        egsum = mp.fsum(eg)
        values = [v/egsum for v in eg]
        return support, values


def support_pmf(nc, ntotal, ngood, nsample):
    """
    Support and PMF values for Fisher's noncentral hypergeometric distr.

    Requires 0 < nc < inf.

    The support of the distribution is k_min <= k <= k_max, where

        k_min = max(0, nsample - (ntotal - ngood))
        k_max = min(nsample, ngood).

    Let

        c_k = choose(ngood, k) * choose(ntotal - ngood, nsample - k) * nc**k

    The probability mass function is

        p_k = c_k / sum(c_i for i in [k_min, k_max])

    Returns
    -------
    sup : range
        The support of the distribution, represented as a Python range.
    pmf : list
        The values of the probability mass function on the support.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.distributions import fishers_noncentral_hypergeometric
    >>> mp.dps = 24
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
    _hg._validate(ntotal, ngood, nsample)
    prec = mp.prec
    support, values = _support_pmf(nc, ntotal, ngood, nsample, prec)
    return support, values


_c_k_formula = r"""
.. math::

   c_k = \\binom{\\textsf{ngood}}{k}\\binom{\\textsf{ntotal} - \\textsf{ngood}}{\\textsf{nsample} - k}\\textsf{nc}^{k}

"""

_kmin_kmax_values = r"""
.. math::

   k_{\\textsf{min}} = \\textrm{max}(0, \\textsf{nsample} - (\\textsf{ntotal} - \\textsf{ngood}))

.. math::

   k_{\\textsf{max}} = \\textrm{min}(\\textsf{nsample}, \\textsf{ngood}).

"""

_p_k_formula = r"""
.. math::

   p_k = \\frac{c_k}
               {\\sum_{j=k_{\\textsf{min}}}^{k_{\\textsf{max}}} c_j}

"""

support_pmf._docstring_re_subs = [
    (' inf[.]', r':math:`\\infty`.', 0, 0),
    (r'c_k.*\*\*k', _c_k_formula, 0, 0),
    ('k_min <= k <= k_max',
     r':math:`k_{\\textsf{min}} \\le k \\le k_{\\textsf{max}}`', 0, 0),
    (r'    k_min =.*ngood\)\.', _kmin_kmax_values, 0, re.DOTALL),
    (r'    p_k =.*max\]\)', _p_k_formula, 0, 0),
]


def pmf_dict(nc, ntotal, ngood, nsample):
    """
    PMF as a dictionary.
    """
    _hg._validate(ntotal, ngood, nsample)
    sup, values = support_pmf(nc, ntotal, ngood, nsample)
    return dict(zip(sup, values))


def pmf(k, nc, ntotal, ngood, nsample):
    """
    PMF of Fisher's noncentral hypergeometric distribution.
    """
    _hg._validate(ntotal, ngood, nsample)
    sup, p = support_pmf(nc, ntotal, ngood, nsample)
    if k in sup:
        return p[k - sup[0]]
    else:
        return mp.zero


def cdf(k, nc, ntotal, ngood, nsample):
    """
    CDF of Fisher's noncentral hypergeometric distribution.
    """
    _hg._validate(ntotal, ngood, nsample)
    sup, p = support_pmf(nc, ntotal, ngood, nsample)
    if k < sup[0]:
        return mp.zero
    elif k >= sup[-1]:
        return mp.one
    else:
        return mp.fsum(p[:k - sup[0] + 1])


def sf(k, nc, ntotal, ngood, nsample):
    """
    Survival function of Fisher's noncentral hypergeometric distribution.
    """
    _hg._validate(ntotal, ngood, nsample)
    sup, p = support_pmf(nc, ntotal, ngood, nsample)
    if k < sup[0]:
        return mp.one
    elif k >= sup[-1]:
        return mp.zero
    else:
        return mp.fsum(p[k - sup[0] + 1:])


def mode(nc, ntotal, ngood, nsample):
    """
    Mode of Fisher's noncentral hypergeometric distribution.

    In cases where the maximum of the PMF occurs twice, the larger index
    of the two is returned.

    Returns
    -------
    m : mpmath.mp.mpf
        The mode of the distribution.

    Examples
    --------
    >>> from mpsci.distributions import fishers_noncentral_hypergeometric
    >>> fishers_noncentral_hypergeometric.mode(2.5, 16, 8, 10)
    6

    In this example, the PMF is symmetric about the mean, and the maximum
    value of the PMF occurs at k=3 and k=4.  The value 4 is returned for the
    mode.

    >>> fishers_noncentral_hypergeometric.support_pmf(1, 14, 7, 7)
    (range(0, 8),
     [mpf('0.00029137529137529148'),
      mpf('0.014277389277389273'),
      mpf('0.12849650349650352'),
      mpf('0.35693473193473196'),
      mpf('0.35693473193473196'),
      mpf('0.12849650349650352'),
      mpf('0.014277389277389273'),
      mpf('0.00029137529137529148')])
    >>> fishers_noncentral_hypergeometric.mode(1, 14, 7, 7)
    4

    """
    _hg._validate(ntotal, ngood, nsample)
    with mp.extradps(5):
        nc = mp.mpf(nc)
        # Using the notation from the wikipedia page...
        A = nc - 1
        B = ngood + nsample - ntotal - (ngood + nsample + 2)*nc
        C = (ngood + 1)*(nsample + 1)*nc
        m = mp.floor(-2*C / (B - mp.sqrt(B**2 - 4*A*C)))
        return int(m)


def mean(nc, ntotal, ngood, nsample):
    """
    Mean of Fisher's noncentral hypergeometric distribution.

    This calculation is implemented as a weighted sum over the support.
    It may be very slow for large parameters.

    Returns
    -------
    m : mpmath.mp.mpf
        The mean of the distribution.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.distributions import fishers_noncentral_hypergeometric
    >>> mp.dps = 24
    >>> fishers_noncentral_hypergeometric.mean(2.5, 16, 8, 10)
    mpf('5.89685838859408792634258808')
    """
    _hg._validate(ntotal, ngood, nsample)
    sup, p = support_pmf(nc, ntotal, ngood, nsample)
    n = len(p)
    with mp.extradps(5):
        return mp.fsum([sup[k]*p[k] for k in range(n)])
