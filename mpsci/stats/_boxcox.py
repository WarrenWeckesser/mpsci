
import mpmath


__all__ = ['boxcox', 'boxcox1p']


def boxcox(x, lmbda):
    x = mpmath.mpf(x)
    lmbda = mpmath.mpf(lmbda)
    if lmbda == 0:
        return mpmath.log(x)
    else:
        return mpmath.powm1(x, lmbda) / lmbda


def boxcox1p(x, lmbda):
    x = mpmath.mpf(x)
    lmbda = mpmath.mpf(lmbda)
    one = mpmath.mpf(1)
    if lmbda == 0:
        return mpmath.log(one + x)
    else:
        return mpmath.powm1(one + x, lmbda) / lmbda
