
import mpmath
from mpsci.distributions import betaprime


mpmath.mp.dps = 80


# Expected values were computed with Wolfram Alpha, e.g.
#     PDF[BetaPrimeDistribution[2, 7/2], 3] = 189/8192


def test_pdf():
    x = 3.0
    a = 2
    b = 3.5
    p = betaprime.pdf(x, a, b)
    expected = mpmath.mpf('189/8192')
    assert mpmath.almosteq(p, expected)


def test_cdf():
    x = 3.0
    a = 2
    b = 3.5
    p = betaprime.cdf(x, a, b)
    expected = mpmath.mpf('995/1024')
    assert mpmath.almosteq(p, expected)


def test_invcdf():
    p = mpmath.mpf('995/1024')
    a = 2
    b = 3.5
    x = betaprime.invcdf(p, a, b)
    expected = 3
    assert mpmath.almosteq(x, expected)


def test_sf():
    x = 3.0
    a = 2
    b = 3.5
    p = betaprime.sf(x, a, b)
    expected = mpmath.mpf('29/1024')
    assert mpmath.almosteq(p, expected)


def test_invsf():
    p = mpmath.mpf('29/1024')
    a = 2
    b = 3.5
    x = betaprime.invsf(p, a, b)
    expected = 3
    assert mpmath.almosteq(x, expected)


def test_mode():
    a = 2
    b = 3.5
    m = betaprime.mode(a, b)
    expected = mpmath.mpf('2/9')
    assert mpmath.almosteq(m, expected)


def test_mean():
    a = 2
    b = 3.5
    m = betaprime.mean(a, b)
    expected = mpmath.mpf('0.8')
    assert mpmath.almosteq(m, expected)


def test_var():
    a = 2
    b = 3.5
    v = betaprime.var(a, b)
    expected = mpmath.mpf('0.96')
    assert mpmath.almosteq(v, expected)


def test_skewness():
    a = 2
    b = 3.5
    s = betaprime.skewness(a, b)
    expected = 13*mpmath.sqrt(mpmath.mpf('2/3'))
    assert mpmath.almosteq(s, expected)
