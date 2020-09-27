
import mpmath


__all__ = ['logbinomial']


def logbinomial(n, k):
    """
    Natural logarithm of binomial(n, k).

    Examples
    --------
    >>> import mpmath
    >>> mpmath.mp.dps = 25
    >>> from mpsci.func import logbinomial

    Compute the log of C(1500, 450).

    >>> logbinomial(1500, 450)
    mpf('912.5010192350457701746286796')

    Verify that it is the expected value.

    >>> mpmath.log(mpmath.binomial(1500, 450))
    mpf('912.5010192350457701746286773')
    """
    if n < 0:
        raise ValueError('n must be nonnegative')
    if k < 0:
        raise ValueError('k must be nonnegative')
    if k > n:
        raise ValueError('k must not exceed n')

    with mpmath.extradps(5):
        return (mpmath.loggamma(n + 1)
                - mpmath.loggamma(k + 1)
                - mpmath.loggamma(mpmath.fsum([n + 1, -k])))
