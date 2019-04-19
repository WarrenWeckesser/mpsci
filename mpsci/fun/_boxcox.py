
import mpmath


__all__ = ['boxcox', 'boxcox1p']


def boxcox(x, lmbda):
    """
    Box-Cox transformation of x.

    The Box-Cox transformation is::

                      { log(x)          if lmbda == 0,
        f(x; lmbda) = {
                      { x**lmbda - 1
                      { ------------    if lmbda != 0
                      {    lmbda

    """
    x = mpmath.mpf(x)
    lmbda = mpmath.mpf(lmbda)
    if lmbda == 0:
        return mpmath.log(x)
    else:
        return mpmath.powm1(x, lmbda) / lmbda


def boxcox1p(x, lmbda):
    """
    Box-Cox transformation of 1 plus x.

    The transformation is::

                      { log(1+x)            if lmbda == 0,
        f(x; lmbda) = {
                      { (1+x)**lmbda - 1
                      { ----------------    if lmbda != 0
                      {      lmbda

    """
    x = mpmath.mpf(x)
    lmbda = mpmath.mpf(lmbda)
    one = mpmath.mpf(1)
    if lmbda == 0:
        return mpmath.log1p(x)
    else:
        return mpmath.powm1(one + x, lmbda) / lmbda
