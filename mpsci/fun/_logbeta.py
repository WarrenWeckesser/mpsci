
import mpmath


__all__ = ['logbeta']


def logbeta(x, y):
    """
    Log of beta(x, y)
    """
    with mpmath.extradps(5):
        mx = mpmath.mpf(x)
        my = mpmath.mpf(y)
        return mpmath.loggamma(x) + mpmath.loggamma(y) - mpmath.loggamma(mx + my)
