
import mpmath


__all__ = ['yeo_johnson', 'inv_yeo_johnson']


def yeo_johnson(x, lmbda):
    r"""
    Yeo-Johnson transformation of x.

    See https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        lmbda = mpmath.mpf(lmbda)
        if x >= 0:
            if lmbda == 0:
                return mpmath.log1p(x)
            else:
                return mpmath.expm1(lmbda*mpmath.log1p(x))/lmbda
        else:
            if lmbda == 2:
                return -mpmath.log1p(-x)
            else:
                lmb2 = 2 - lmbda
                return -mpmath.expm1(lmb2*mpmath.log1p(-x))/lmb2


def inv_yeo_johnson(x, lmbda):
    """
    Inverse Yeo-Johnson transformation.

    See https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        lmbda = mpmath.mpf(lmbda)
        if x >= 0:
            if lmbda == 0:
                return mpmath.expm1(x)
            else:
                return mpmath.expm1(mpmath.log1p(lmbda*x)/lmbda)
        else:
            if lmbda == 2:
                return -mpmath.expm1(-x)
            else:
                lmb2 = 2 - lmbda
                return -mpmath.expm1(mpmath.log1p(-lmb2*x)/lmb2)
