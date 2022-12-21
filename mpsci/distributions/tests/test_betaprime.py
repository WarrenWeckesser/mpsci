import pytest
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


@pytest.mark.parametrize(
    'x, a, b, p',
    [(3.0, 2.0, 3.5, mpmath.mpf('995/1024')),
     (mpmath.mpf('1e-12'), 1.25, 2.5,
      mpmath.mpf('2.936625089475862805103544292783441344239148068331725458'
                 '237324122914262883428805256602650237662010653e-15')),
     (10**25, 0.5, 0.125,
      mpmath.mpf('0.999355535410004101045571916149514907513532205970073693'
                 '3856311738452896343983734668'))]
)
def test_cdf_invcdf(x, a, b, p):
    p_computed = betaprime.cdf(x, a, b)
    assert mpmath.almosteq(p_computed, p)
    x_computed = betaprime.invcdf(p, a, b)
    assert mpmath.almosteq(x_computed, x)


@pytest.mark.parametrize(
    'x, a, b, p',
    [(3.0, 2.0, 3.5, mpmath.mpf('29/1024')),
     (0.125, 1.25, 2.5,
      mpmath.mpf('0.828718068951133944885573567547680175087963872519064943'
                 '1122121393516469130876181115151511585080223107')),
     (10**25, 0.5, 0.125,
      mpmath.mpf('0.000644464589995898954428083850485092486467794029926306'
                 '61436882615471036560162653318007898'))]
)
def test_sf(x, a, b, p):
    p_computed = betaprime.sf(x, a, b)
    assert mpmath.almosteq(p_computed, p)
    x_computed = betaprime.invsf(p, a, b)
    assert mpmath.almosteq(x_computed, x)


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
