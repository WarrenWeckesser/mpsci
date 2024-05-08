"""
Noncentral t distribution
-------------------------

Currently only the shape parameters (degrees of freedom `df` and
noncentrality `nc`) are implemented.

"""

from functools import lru_cache
from mpmath import mp
from ._common import _validate_moment_n


__all__ = ['pdf', 'logpdf', 'support', 'mean', 'var', 'noncentral_moment']


def pdf(x, df, nc):
    """
    Probability density function of the noncentral t distribution.

    The infinite series is estimated with `mpmath.nsum`.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        df = mp.mpf(df)
        nc = mp.mpf(nc)

        if x == 0:
            logp = (-nc**2/2
                    - mp.log(mp.pi)/2
                    - mp.log(df)/2
                    + mp.loggamma((df + 1)/2)
                    - mp.loggamma(df/2))
            p = mp.exp(logp)
        else:
            logc = (df*mp.log(df)/2
                    - nc**2/2
                    - mp.loggamma(df/2)
                    - mp.log(mp.pi)/2
                    - (df + 1)/2 * mp.log(df + x**2))
            c = mp.exp(logc)

            def _pdf_term(i):
                logterm = (mp.loggamma((df + i + 1)/2)
                           + i*mp.log(x*nc)
                           + i*mp.log(2/(df + x**2))/2
                           - mp.loggamma(i + 1))
                return mp.exp(logterm).real

            s = mp.nsum(_pdf_term, [0, mp.inf])
            p = c * s
        return p


def logpdf(x, df, nc):
    """
    Logarithm of the PDF of the noncentral t distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        df = mp.mpf(df)
        nc = mp.mpf(nc)

        if x == 0:
            logp = (-nc**2/2
                    - mp.log(mp.pi)/2
                    - mp.log(df)/2
                    + mp.loggamma((df + 1)/2)
                    - mp.loggamma(df/2))
        else:
            logc = (df*mp.log(df)/2
                    - nc**2/2
                    - mp.loggamma(df/2)
                    - mp.log(mp.pi)/2
                    - (df + 1)/2 * mp.log(df + x**2))

            def _pdf_term(i):
                logterm = (mp.loggamma((df + i + 1)/2)
                           + i*mp.log(x*nc)
                           + i*mp.log(2/(df + x**2))/2
                           - mp.loggamma(i + 1))
                return mp.exp(logterm).real

            s = mp.nsum(_pdf_term, [0, mp.inf])
            logp = logc + mp.log(s)
        return logp


def support(df, nc):
    """
    Support of the noncentral t distribution.
    """
    with mp.extradps(5):
        return (mp.ninf, mp.inf)


def mean(df, nc):
    """
    Mean of the noncentral t distribution.
    """
    # XXX Require df > 1.
    with mp.extradps(5):
        df = mp.mpf(df)
        nc = mp.mpf(nc)
        logm = (mp.log(nc)
                + mp.log(df/2)/2
                + mp.loggamma((df - 1)/2)
                - mp.loggamma(df/2))
        return mp.exp(logm)


def var(df, nc):
    """
    Variance of the noncentral t distribution.
    """
    # XXX Require df > 2.
    with mp.extradps(5):
        df = mp.mpf(df)
        nc = mp.mpf(nc)
        c = mp.exp(mp.loggamma((df - 1)/2) - mp.loggamma(df/2))
        return df/(df - 2) * (1 + nc**2) - df/2 * nc**2 * c**2


@lru_cache
def _poly_coeffs(k):
    """
    Generate the coefficients of the polynomial that is the result of
    the expression exp(-x**2/2) * (d^k/dx^k)exp(x**2/2).

    The coefficients are returned in increasing order of the power.
    That is, the return value [c0, c1, c2, c3] represents the polynomial
    c0 + c1*x + c2*x**2 + c3*x**3.
    """
    if k == 0:
        return [1]
    c = [0, 1]
    for _ in range(2, k+1):
        c = [a + b for a, b in zip([i*j for i, j in enumerate(c[1:], start=1)],
                                   [0] + c[:-2])]
        c.extend([0, 1])
    return c


def noncentral_moment(n, df, nc):
    """
    Noncentral moment (i.e. raw moment) for the noncentral t distribution.

    n is the order of the moment to be computed.  The moment is only
    defined for n < df.  If n >= df, nan is returned.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        df = mp.mpf(df)
        nc = mp.mpf(nc)
        if df <= n:
            return mp.nan
        c = _poly_coeffs(n)
        return (mp.exp((n/2)*mp.log(df/2)
                       + mp.loggamma((df - n)/2)
                       - mp.loggamma(df/2))
                * mp.polyval(c[::-1], nc))
