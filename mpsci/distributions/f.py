"""
F distribution
--------------

"""

import mpmath
from ..fun import logbeta, xlogy, xlog1py


__all__ = ['pdf', 'logpdf', 'cdf']


def pdf(x, dfn, dfd):
    """
    Probability density function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.mp.extradps(5):
        x = mpmath.mp.mpf(x)

        dfn = mpmath.mp.mpf(dfn)
        dfd = mpmath.mp.mpf(dfd)
        r = dfn / dfd
        hdfn = dfn / 2
        hdfd = dfd / 2
        p = (r**hdfn
             * x**(hdfn - 1)
             * (1 + r*x)**(-(hdfn + hdfd))
             / mpmath.beta(hdfn, hdfd))
        return p


def logpdf(x, dfn, dfd):
    """
    Natural log. of the probability density function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    if x <= 0:
        return -mpmath.inf
    with mpmath.mp.extradps(5):
        x = mpmath.mp.mpf(x)
        if x <= 0:
            return mpmath.mp.zero
        dfn = mpmath.mp.mpf(dfn)
        dfd = mpmath.mp.mpf(dfd)
        r = dfn / dfd
        hdfn = dfn / 2
        hdfd = dfd / 2
        lp = (hdfn * (mpmath.log(dfn) - mpmath.log(dfd))
              + xlogy(hdfn - 1, x)
              - xlog1py(hdfn + hdfd, r*x)
              - logbeta(hdfn, hdfd))
        return lp


def cdf(x, dfn, dfd):
    """
    Cumulative distribution function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.mp.extradps(5):
        x = mpmath.mp.mpf(x)
        dfn = mpmath.mp.mpf(dfn)
        dfd = mpmath.mp.mpf(dfd)
        dfnx = dfn * x
        return mpmath.betainc(dfn/2, dfd/2, x2=dfnx/(dfnx + dfd),
                              regularized=True)
