"""
Student's t distribution
------------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf']


def logpdf(x, df):
    """
    Logarithm of the PDF of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        df = mpmath.mpf(df)
        h = (df + 1) / 2
        logp = (mpmath.loggamma(h)
                - mpmath.log(df * mpmath.pi)/2
                - mpmath.loggamma(df/2)
                - h * mpmath.log1p(x**2/df))
    return logp


def pdf(x, df):
    """
    PDF of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    return mpmath.exp(logpdf(x, df))


def cdf(x, df):
    """
    CDF of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mpmath.extradps(5):
        half = mpmath.mp.one/2
        x = mpmath.mpf(x)
        df = mpmath.mpf(df)
        h = (df + 1) / 2
        p1 = x * mpmath.gamma(h)
        p2 = mpmath.hyp2f1(half, h, 3*half, -x**2/df)
        return half + p1*p2/mpmath.sqrt(mpmath.pi*df)/mpmath.gamma(df/2)


def sf(x, df):
    """
    Survival function of Student's t distribution.
    """
    if df <= 0:
        raise ValueError('df must be greater than 0')

    with mpmath.extradps(5):
        half = mpmath.mp.one/2
        x = mpmath.mpf(x)
        df = mpmath.mpf(df)
        h = (df + 1) / 2
        p1 = x * mpmath.gamma(h)
        p2 = mpmath.hyp2f1(half, h, 3*half, -x**2/df)
        return half - p1*p2/mpmath.sqrt(mpmath.pi*df)/mpmath.gamma(df/2)


def invcdf(p, df):
    """
    Inverse of the CDF for Student's t distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    if df <= 0:
        raise ValueError('df must be greater than 0')

    if p == 0:
        return mpmath.ninf
    if p == 1:
        return mpmath.inf

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        df = mpmath.mpf(df)

        def _func(x):
            return cdf(x, df) - p

        x = mpmath.findroot(_func, 0)

    return x


def invsf(p, df):
    """
    Inverse of the survival function for Student's t distribution.
    """
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    if df <= 0:
        raise ValueError('df must be greater than 0')

    if p == 0:
        return mpmath.inf
    if p == 1:
        return mpmath.ninf

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        df = mpmath.mpf(df)

        def _func(x):
            return sf(x, df) - p

        x = mpmath.findroot(_func, 0)

    return x
