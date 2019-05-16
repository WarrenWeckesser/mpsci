
import mpmath


__all__ = ['logbinomial']


def logbinomial(n, k):
    """
    Natural logarithm of binomial(n, k).
    """
    # XXX To do: validate arguments.

    with mpmath.extradps(5):
        return (mpmath.loggamma(n + 1)
                - mpmath.loggamma(k + 1)
                - mpmath.loggamma(mpmath.fsum([n + 1, -k])))
