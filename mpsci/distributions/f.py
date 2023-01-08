"""
F distribution
--------------

"""

from mpmath import mp
from ..fun import logbeta, xlogy, xlog1py


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'var']


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
