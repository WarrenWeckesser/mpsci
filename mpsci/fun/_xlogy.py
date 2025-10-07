
from mpmath import mp


__all__ = ['xlogy', 'xlog1py']


def xlogy(x, y):
    """
    x*log(y)

    If x is 0 and y is not nan, 0 is returned.

    *See also:* :func:`xlog1py`
    """
    if mp.isnan(x) or mp.isnan(y):
        return mp.nan
    if x == 0:
        return mp.zero
    else:
        return x * mp.log(y)


xlogy._docstring_re_subs = [
    (r'x\*log\(y\)', r':math:`x\\textrm{log}(y)`', 0, 0),
]


def xlog1py(x, y):
    """
    x*log(1+y)

    If ``x`` is 0 and ``y`` is not ``nan``, 0 is returned.

    This function is mathematically equivalent to ``mpsci.fun.xlogy(1 + y)``.
    It avoids the loss of precision that can result if ``y`` is very small.

    *See also:* :func:`xlogy`
    """
    if mp.isnan(x) or mp.isnan(y):
        return mp.nan
    if x == 0:
        return mp.zero
    else:
        return x * mp.log1p(y)


xlog1py._docstring_re_subs = [
    (r'x\*log\(1\+y\)', r':math:`x\\textrm{log}(1+y)`', 0, 0),
]
