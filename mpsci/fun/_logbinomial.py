from mpmath import mp


__all__ = ['logbinomial']


def logbinomial(n, k):
    """
    Natural logarithm of binomial(n, k).

    Examples
    --------
    >>> from mpmath import mp
    >>> mp.dps = 25
    >>> from mpsci.fun import logbinomial

    Compute the log of C(1500, 450).

    >>> logbinomial(1500, 450)
    mpf('912.5010192350457701746286796')

    Verify that it is the expected value.

    >>> mp.log(mp.binomial(1500, 450))
    mpf('912.5010192350457701746286773')
    """
    if n < 0:
        raise ValueError('n must be nonnegative')
    if k < 0:
        raise ValueError('k must be nonnegative')
    if k > n:
        raise ValueError('k must not exceed n')

    with mp.extradps(5):
        return (mp.loggamma(n + 1)
                - mp.loggamma(k + 1)
                - mp.loggamma(mp.fsum([n + 1, -k])))
