"""
Noncentral F distribution
-------------------------
"""

import mpmath as _mp
from ..fun import logbeta as _logbeta


__all__ = ['pdf', 'cdf', 'mean', 'var']


def _pdf_term(k, x, dfn, dfd, nc):
    halfnc = nc / 2
    halfdfn = dfn / 2
    halfdfd = dfd / 2
    logr = (-halfnc
            + k*_mp.log(halfnc)
            - _logbeta(halfdfd, halfdfn + k)
            - _mp.loggamma(k + 1)
            + (halfdfn + k) * (_mp.log(dfn) - _mp.log(dfd))
            + (halfdfn + halfdfd + k) * (_mp.log(dfd) -
                                         _mp.log(dfd + dfn*x))
            + (halfdfn - 1 + k) * _mp.log(x))
    return _mp.exp(logr)


def pdf(x, dfn, dfd, nc):
    """
    PDF of the noncentral F distribution.
    """
    if x < 0:
        return _mp.mp.zero

    def _pdfk(k):
        return _pdf_term(k, x, dfn, dfd, nc)

    with _mp.extradps(5):
        x = _mp.mpf(x)
        dfn = _mp.mpf(dfn)
        dfd = _mp.mpf(dfd)
        nc = _mp.mpf(nc)
        p = _mp.nsum(_pdfk, [0, _mp.inf])
        return p


def _cdf_term(k, x, dfn, dfd, nc):
    halfnc = nc / 2
    halfdfn = dfn / 2
    halfdfd = dfd / 2
    log_coeff = _mp.fsum([k*_mp.log(halfnc), -halfnc,
                          -_mp.loggamma(k + 1)])
    coeff = _mp.exp(log_coeff)
    r = coeff * _mp.betainc(a=halfdfn + k, b=halfdfd,
                            x1=0, x2=dfn*x/(dfd + dfn*x), regularized=True)
    return r


def cdf(x, dfn, dfd, nc):
    """
    CDF of the noncentral F distribution.
    """
    if x < 0:
        return _mp.mp.zero

    def _cdfk(k):
        return _cdf_term(k, x, dfn, dfd, nc)

    with _mp.extradps(5):
        x = _mp.mpf(x)
        dfn = _mp.mpf(dfn)
        dfd = _mp.mpf(dfd)
        nc = _mp.mpf(nc)
        p = _mp.nsum(_cdfk, [0, _mp.inf])
        return p


def mean(dfn, dfd, nc):
    """
    Mean of the noncentral F distribution.
    """
    if dfd <= 2:
        return _mp.mp.nan

    with _mp.extradps(5):
        nc = _mp.mpf(nc)
        dfn = _mp.mpf(dfn)
        dfd = _mp.mpf(dfd)
        return dfd * (dfn + nc) / dfn / (dfd - 2)


def var(dfn, dfd, nc):
    """
    Variance of the noncentral F distribution.
    """
    if dfd <= 4:
        return _mp.mp.nan

    with _mp.extradps(5):
        nc = _mp.mpf(nc)
        dfn = _mp.mpf(dfn)
        dfd = _mp.mpf(dfd)
        v = (2*((dfn + nc)**2 +
                (dfn + 2*nc) * (dfd - 2)) /
               ((dfd - 2)**2 * (dfd - 4)) *
             (dfd/dfn)**2)
        return v
