"""
Noncentral F distribution
-------------------------
"""

from mpmath import mp
from ..fun import logbeta as _logbeta


__all__ = ['pdf', 'cdf', 'support', 'mean', 'var']


def _pdf_term(k, x, dfn, dfd, nc):
    halfnc = nc / 2
    halfdfn = dfn / 2
    halfdfd = dfd / 2
    logr = (-halfnc
            + k*mp.log(halfnc)
            - _logbeta(halfdfd, halfdfn + k)
            - mp.loggamma(k + 1)
            + (halfdfn + k) * (mp.log(dfn) - mp.log(dfd))
            + (halfdfn + halfdfd + k) * (mp.log(dfd) -
                                         mp.log(dfd + dfn*x))
            + (halfdfn - 1 + k) * mp.log(x))
    return mp.exp(logr)


def pdf(x, dfn, dfd, nc):
    """
    PDF of the noncentral F distribution.
    """
    if x < 0:
        return mp.zero

    def _pdfk(k):
        return _pdf_term(k, x, dfn, dfd, nc)

    with mp.extradps(5):
        x = mp.mpf(x)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        nc = mp.mpf(nc)
        p = mp.nsum(_pdfk, [0, mp.inf])
        return p


def _cdf_term(k, x, dfn, dfd, nc):
    halfnc = nc / 2
    halfdfn = dfn / 2
    halfdfd = dfd / 2
    log_coeff = mp.fsum([k*mp.log(halfnc), -halfnc,
                         -mp.loggamma(k + 1)])
    coeff = mp.exp(log_coeff)
    r = coeff * mp.betainc(a=halfdfn + k, b=halfdfd,
                           x1=0, x2=dfn*x/(dfd + dfn*x), regularized=True)
    return r


def cdf(x, dfn, dfd, nc):
    """
    CDF of the noncentral F distribution.
    """
    if x < 0:
        return mp.zero

    def _cdfk(k):
        return _cdf_term(k, x, dfn, dfd, nc)

    with mp.extradps(5):
        x = mp.mpf(x)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        nc = mp.mpf(nc)
        p = mp.nsum(_cdfk, [0, mp.inf])
        return p


def support(dfn, dfd, nc):
    """
    Support of the noncentral F distribution.
    """
    with mp.extradps(5):
        nc = mp.mpf(nc)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        return (mp.zero, mp.inf)


def mean(dfn, dfd, nc):
    """
    Mean of the noncentral F distribution.
    """
    if dfd <= 2:
        return mp.nan

    with mp.extradps(5):
        nc = mp.mpf(nc)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        return dfd * (dfn + nc) / dfn / (dfd - 2)


def var(dfn, dfd, nc):
    """
    Variance of the noncentral F distribution.
    """
    if dfd <= 4:
        return mp.nan

    with mp.extradps(5):
        nc = mp.mpf(nc)
        dfn = mp.mpf(dfn)
        dfd = mp.mpf(dfd)
        v = (2*((dfn + nc)**2 +
                (dfn + 2*nc) * (dfd - 2)) /
               ((dfd - 2)**2 * (dfd - 4)) *
             (dfd/dfn)**2)
        return v
