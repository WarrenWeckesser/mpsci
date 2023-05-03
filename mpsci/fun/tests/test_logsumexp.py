from mpmath import mp
from mpsci.fun import logsumexp


@mp.workdps(50)
def test_basic_logsumexp():
    x = [1.0, -1.0, -2.0]
    y = logsumexp(x)
    expected = mp.log(mp.fsum([mp.exp(xi) for xi in x]))
    assert mp.almosteq(y, expected)


@mp.workdps(50)
def test_weighted_logsumexp():
    x = [1.0, 0.5, -1.0, -2.0]
    w = [3.5, 0.0,  1.0,  3.0]
    y = logsumexp(x, weights=w)
    wsum = mp.fsum([wi*mp.exp(xi) for xi, wi in zip(x, w)])
    expected = mp.log(wsum)
    assert mp.almosteq(y, expected)
