
import mpmath
from mpsci.fun import logsumexp


def test_basic_logsumexp():
    x = [1.0, -1.0, -2.0]
    y = logsumexp(x)
    expected = mpmath.log(mpmath.fsum([mpmath.exp(xi) for xi in x]))
    assert mpmath.almosteq(y, expected)


def test_weighted_logsumexp():
    x = [1.0, 0.5, -1.0, -2.0]
    w = [3.5, 0.0,  1.0,  3.0]
    y = logsumexp(x, weights=w)
    wsum = mpmath.fsum([wi*mpmath.exp(xi) for xi, wi in zip(x, w)])
    expected = mpmath.log(wsum)
    assert mpmath.almosteq(y, expected)
