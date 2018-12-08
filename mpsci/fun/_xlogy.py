
from mpmath import mp


__all__ = ['xlogy']


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
