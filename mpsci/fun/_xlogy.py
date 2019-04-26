
from mpmath import mp


__all__ = ['xlogy', 'xlog1py']


def xlogy(x, y):
    """
    x*log(y)

    If x is 0 and y is not nan, 0 is returned.
    """
    if mp.isnan(x) or mp.isnan(y):
        return mp.nan
    if x == 0:
        return mp.zero
    else:
        return x * mp.log(y)


def xlog1py(x, y):
    """
    x*log(1+y)

    If x is 0 and y is not nan, 0 is returned.

    This function is mathematically equivalent to `xlogy(1 + y)`.  It avoids
    the loss of precision that can result if y is very small.
    """
    if mp.isnan(x) or mp.isnan(y):
        return mp.nan
    if x == 0:
        return mp.zero
    else:
        return x * mp.log1p(y)
