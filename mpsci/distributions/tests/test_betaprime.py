import pytest
from mpmath import mp
from mpsci.distributions import betaprime


# Expected values were computed with Wolfram Alpha, e.g.
#     PDF[BetaPrimeDistribution[2, 7/2], 3] = 189/8192


def test_pdf():
    with mp.workdps(80):
        x = 3.0
        a = 2
        b = 3.5
        p = betaprime.pdf(x, a, b)
        expected = mp.mpf('189/8192')
        assert mp.almosteq(p, expected)


@pytest.mark.parametrize(
    'x, a, b, p',
    [(3.0, 2.0, 3.5, '995/1024'),
     ('1e-12', 1.25, 2.5,
      '2.936625089475862805103544292783441344239148068331725458'
      '237324122914262883428805256602650237662010653e-15'),
     (10**25, 0.5, 0.125,
      '0.999355535410004101045571916149514907513532205970073693'
      '3856311738452896343983734668')]
)
def test_cdf_invcdf(x, a, b, p):
    with mp.workdps(80):
        x = mp.mpf(x)
        a = mp.mpf(a)
        b = mp.mpf(b)
        p = mp.mpf(p)
        p_computed = betaprime.cdf(x, a, b)
        assert mp.almosteq(p_computed, p)
        x_computed = betaprime.invcdf(p, a, b)
        assert mp.almosteq(x_computed, x, rel_eps=mp.mpf('1e-77'))


@pytest.mark.parametrize(
    'x, a, b, p',
    [(3.0, 2.0, 3.5, '29/1024'),
     (0.125, 1.25, 2.5,
      '0.828718068951133944885573567547680175087963872519064943'
      '1122121393516469130876181115151511585080223107'),
     (10**25, 0.5, 0.125,
      '0.000644464589995898954428083850485092486467794029926306'
      '61436882615471036560162653318007898')]
)
def test_sf(x, a, b, p):
    with mp.workdps(80):
        x = mp.mpf(x)
        a = mp.mpf(a)
        b = mp.mpf(b)
        p = mp.mpf(p)
        p_computed = betaprime.sf(x, a, b)
        assert mp.almosteq(p_computed, p)
        x_computed = betaprime.invsf(p, a, b)
        assert mp.almosteq(x_computed, x)


def test_mode():
    with mp.workdps(80):
        a = 2
        b = 3.5
        m = betaprime.mode(a, b)
        expected = mp.mpf('2/9')
        assert mp.almosteq(m, expected)


def test_mean():
    with mp.workdps(80):
        a = 2
        b = 3.5
        m = betaprime.mean(a, b)
        expected = mp.mpf('0.8')
        assert mp.almosteq(m, expected)


def test_var():
    with mp.workdps(80):
        a = 2
        b = 3.5
        v = betaprime.var(a, b)
        expected = mp.mpf('0.96')
        assert mp.almosteq(v, expected)


def test_skewness():
    with mp.workdps(80):
        a = 2
        b = 3.5
        s = betaprime.skewness(a, b)
        expected = 13*mp.sqrt(mp.mpf('2/3'))
        assert mp.almosteq(s, expected)
