"""
F distribution
--------------

"""

from mpmath import mp
from ._common import _validate_p
from ..fun import logbeta, xlogy, xlog1py, betaincinv


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'entropy']


def pdf(x, dfn, dfd):
    """
    Probability density function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        r = dfn / dfd
        hdfn = dfn / 2
        hdfd = dfd / 2
        p = (r**hdfn
             * x**(hdfn - 1)
             * (1 + r*x)**(-(hdfn + hdfd))
             / mp.beta(hdfn, hdfd))
        return p


def logpdf(x, dfn, dfd):
    """
    Natural log. of the probability density function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """

    with mp.extradps(5):
        x = mp.mpf(x)
        if x <= 0:
            return -mp.inf
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        r = dfn / dfd
        hdfn = dfn / 2
        hdfd = dfd / 2
        lp = (hdfn * (mp.log(dfn) - mp.log(dfd))
              + xlogy(hdfn - 1, x)
              - xlog1py(hdfn + hdfd, r*x)
              - logbeta(hdfn, hdfd))
        return lp


def cdf(x, dfn, dfd):
    """
    Cumulative distribution function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        dfnx = dfn * x
        return mp.betainc(dfn/2, dfd/2, x2=dfnx/(dfnx + dfd),
                          regularized=True)


def invcdf(p, dfn, dfd):
    """
    Inverse of the CDF of the F distriution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        g = betaincinv(dfn/2, dfd/2, p, method='bisect')
        # XXX Possible loss of precision in (1 - g):
        return (dfd/dfn)*g/(1 - g)


def sf(x, dfn, dfd):
    """
    Survival function of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        dfnx = dfn * x
        return mp.betainc(dfn/2, dfd/2, x1=dfnx/(dfnx + dfd),
                          regularized=True)


def invsf(p, dfn, dfd):
    """
    Inverse of the survival function of the F distriution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        g = betaincinv(dfd/2, dfn/2, p, method='bisect')
        # XXX Possible loss of precison in (1 - g):
        return (dfd/dfn)*(1 - g)/g


def mean(dfn, dfd):
    """
    Mean of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.

    For a finite mean, `dfd` must be greater than 2.
    If `dfd` is less than or equal to 2, this function returns `inf`.
    """
    with mp.extradps(5):
        if dfd > 2:
            dfd = mp.mpf(dfd)
            return dfd/(dfd - 2)
        return mp.inf


def var(dfn, dfd):
    """
    Variance of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.

    For a finite variance, `dfd` must be greater than 4.
    If `dfd` is less than or equal to 4, this function returns `inf`.
    """
    with mp.extradps(5):
        if dfd > 4:
            dfn = mp.mpf(dfn)
            dfd = mp.mpf(dfd)
            return (2 * dfd**2 * (dfn + dfd - 2)
                    / (dfn * (dfd - 2)**2 * (dfd - 4)))
        return mp.inf


def entropy(dfn, dfd):
    """
    Differential entropy of the F distribution.

    `dfn` and `dfd` are the numerator and denominator degrees of freedom, resp.

    """
    with mp.extradps(5):
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        dfn2 = dfn/2
        dfd2 = dfd/2
        dfmean = dfn2 + dfd2
        return (mp.log(dfd/dfn)
                + logbeta(dfn2, dfd2)
                + (1 - dfn2)*mp.digamma(dfn2)
                - (1 + dfd2)*mp.digamma(dfd2)
                + dfmean*mp.digamma(dfmean))
