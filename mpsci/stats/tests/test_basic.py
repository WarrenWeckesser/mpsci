
import mpmath
from mpsci.stats import mean, var, std, variation, gmean, hmean, pmean


# XXX In some of these tests, equality is asserted even though the
# calculation uses floating point.  That these tests currently pass
# might just be luck.

mpmath.mp.dps = 50


def test_mean():
    assert mean([1, 2, 3]) == 2
    assert mean([1, 2, 3], weights=[3, 1, 3]) == 2


def test_var():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(var([2, 4, 6]), mpmath.mpf('8/3'))
        assert mpmath.almosteq(var([2, 4, 6], ddof=1), 4)


def test_std():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(std([2, 4, 6]), mpmath.sqrt(mpmath.mpf('8/3')))
        assert mpmath.almosteq(std([2, 4, 6], ddof=1), 2)


def test_gmean():
    assert mpmath.almosteq(gmean([3, 3**3, 3**5]), 27)


def test_gmean_with_0():
    assert gmean([3, 1, 0, 9]) == 0


def test_gmean_weights_all_one():
    x = [3, 4, 5, 10]
    assert mpmath.almosteq(gmean(x, weights=[1]*len(x)),
                           gmean(x))


def test_gmean_weights():
    x = [2, 3, 5, 8]
    w = [1, 2, 3, 4]
    wgm1 = gmean(x, weights=w)
    wx = sum([[xi]*wi for (xi, wi) in zip(x, w)], [])
    wgm2 = gmean(wx)
    assert mpmath.almosteq(wgm1, wgm2)


def test_gmean_0_weight():
    x = [2, 3, 5, 8, 13, 21]
    w = [1, 3, 5, 0,  4,  0]
    wgm1 = gmean(x, weights=w)
    # Filter out points where the weight is 0.
    xx, ww = zip(*[(xi, wi) for (xi, wi) in zip(x, w) if wi != 0])
    wgm2 = gmean(xx, weights=ww)
    assert mpmath.almosteq(wgm1, wgm2)


def test_hmean():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(hmean([1, 2, 16]), mpmath.mpf('48/25'))


def test_pmean():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(pmean([3, 4, 5], 3), 72**mpmath.mpf('1/3'))
        assert mpmath.almosteq(pmean([2, 2, 1], -2), mpmath.sqrt(2))
        assert pmean([4, 2, 5, 3], mpmath.inf) == 5
        assert pmean([4, 2, 5, 3], -mpmath.inf) == 2


def test_variation0():
    x = [1, 1, 4]
    v = variation(x, ddof=0)
    assert mpmath.almosteq(v**2, mpmath.mpf('0.5'))


def test_variation1():
    x = [1, 1, 4]
    v = variation(x, ddof=1)
    assert mpmath.almosteq(v**2, mpmath.mpf('0.75'))
