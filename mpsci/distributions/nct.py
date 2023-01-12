"""
Noncentral t distribution
-------------------------

Currently only the shape parameters (degrees of freedom `df` and
noncentrality `nc`) are implemented.

"""

from mpmath import mp


__all__ = ['pdf', 'mean', 'var']


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
