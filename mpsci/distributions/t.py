"""
Student's t distribution
------------------------
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'entropy']


def logpdf(x, df):
    """
    Logarithm of the PDF of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mp.extradps(5):
        x = mp.mpf(x)
        df = mp.mpf(df)
        h = (df + 1) / 2
        logp = (mp.loggamma(h)
                - mp.log(df * mp.pi)/2
                - mp.loggamma(df/2)
                - h * mp.log1p(x**2/df))
    return logp


def pdf(x, df):
    """
    PDF of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    return mp.exp(logpdf(x, df))


def cdf(x, df):
    """
    CDF of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mp.extradps(5):
        half = mp.one/2
        x = mp.mpf(x)
        df = mp.mpf(df)
        h = (df + 1) / 2
        p1 = x * mp.gamma(h)
        p2 = mp.hyp2f1(half, h, 3*half, -x**2/df)
        return half + p1*p2/mp.sqrt(mp.pi*df)/mp.gamma(df/2)


def sf(x, df):
    """
    Survival function of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mp.extradps(5):
        half = mp.one/2
        x = mp.mpf(x)
        df = mp.mpf(df)
        h = (df + 1) / 2
        p1 = x * mp.gamma(h)
        p2 = mp.hyp2f1(half, h, 3*half, -x**2/df)
        return half - p1*p2/mp.sqrt(mp.pi*df)/mp.gamma(df/2)


def invcdf(p, df):
    """
    Inverse of the CDF for Student's t distribution.

    This function is also known as the quantile function or the percent
    point function.

    For values far in the tails of the distribution, the solution might
    not be accurate.  Check the results, and increase the precision of
    the calculation if necessary.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mp.ninf
        if p == 1:
            return mp.inf
        if p > 0.5:
            p0 = mp.one - p
        else:
            p0 = p
        df = mp.mpf(df)

        x0 = -mp.one
        count = 0
        while cdf(x0, df) > p0:
            x0 = 2*x0
            count += 1
            if count > mp.prec / 2:
                raise RuntimeError('failed to find a bracketing interval')

        def _func(x):
            return cdf(x, df) - p0

        x = mp.findroot(_func, (x0, x0/2), solver='anderson')
        if p > 0.5:
            x = -x

    return x


def invsf(p, df):
    """
    Inverse of the survival function for Student's t distribution.

    For values far in the tails of the distribution, the solution might
    not be accurate.  Check the results, and increase the precision of
    the calculation if necessary.
    """
    p = _validate_p(p)
    if df <= 0:
        raise ValueError('df must be greater than 0')

    return -invcdf(p, df)


def entropy(df):
    """
    Entropy of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mp.extradps(5):
        df = mp.mpf(df)
        h = df/2
        h1 = (df + 1)/2
        return (h1*(mp.digamma(h1) - mp.digamma(h)) +
                mp.log(mp.sqrt(df)*mp.beta(h, 0.5)))
