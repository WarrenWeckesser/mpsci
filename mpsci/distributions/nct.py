"""
Noncentral t distribution
-------------------------

Currently only the shape parameters (degrees of freedom `df` and
noncentrality `nc`) are implemented.

"""

import mpmath


__all__ = ['pdf', 'mean', 'var']


def pdf(x, df, nc):
    """
    Probability density function of the noncentral t distribution.

    The infinite series is estimated with `mpmath.nsum`.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        df = mpmath.mpf(df)
        nc = mpmath.mpf(nc)

        if x == 0:
            logp = (-nc**2/2
                    - mpmath.log(mpmath.pi)/2
                    - mpmath.log(df)/2
                    + mpmath.loggamma((df + 1)/2)
                    - mpmath.loggamma(df/2))
            p = mpmath.exp(logp)
        else:
            logc = (df*mpmath.log(df)/2
                    - nc**2/2
                    - mpmath.loggamma(df/2)
                    - mpmath.log(mpmath.pi)/2
                    - (df + 1)/2 * mpmath.log(df + x**2))
            c = mpmath.exp(logc)

            def _pdf_term(i):
                logterm = (mpmath.loggamma((df + i + 1)/2)
                           + i*mpmath.log(x*nc)
                           + i*mpmath.log(2/(df + x**2))/2
                           - mpmath.loggamma(i + 1))
                return mpmath.exp(logterm).real

            s = mpmath.nsum(_pdf_term, [0, mpmath.inf])
            p = c * s
        return p


def mean(df, nc):
    """
    Mean of the noncentral t distribution.
    """
    # XXX Require df > 1.
    with mpmath.extradps(5):
        df = mpmath.mpf(df)
        nc = mpmath.mpf(nc)
        logm = (mpmath.log(nc)
                + mpmath.log(df/2)/2
                + mpmath.loggamma((df - 1)/2)
                - mpmath.loggamma(df/2))
        return mpmath.exp(logm)


def var(df, nc):
    """
    Variance of the noncentral t distribution.
    """
    # XXX Require df > 2.
    with mpmath.extradps(5):
        df = mpmath.mpf(df)
        nc = mpmath.mpf(nc)
        c = mpmath.exp(mpmath.loggamma((df - 1)/2) - mpmath.loggamma(df/2))
        return df/(df - 2) * (1 + nc**2) - df/2 * nc**2 * c**2
