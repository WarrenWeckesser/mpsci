
import mpmath
from mpsci.distributions import gauss_kuzmin


mpmath.mp.dps = 40


def test_pmf():
    one = mpmath.mp.one
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.pmf(k)
        expected = -mpmath.log1p(-one/(k + one)**2)/mpmath.log(2)
        assert mpmath.almosteq(p, expected)


def test_logpmf():
    one = mpmath.mp.one
    for k in [1, 2, 3, 9]:
        logp = gauss_kuzmin.logpmf(k)
        expected = mpmath.log(-mpmath.log1p(-one/(k + one)**2)/mpmath.log(2))
        assert mpmath.almosteq(logp, expected)


def test_cdf():
    one = mpmath.mp.one
    two = mpmath.mpf(2)
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.cdf(k)
        assert mpmath.almosteq(p, 1 - mpmath.log((k + two)/(k + one), b=2))


def test_cdf_invcdf_roundtrip():
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.cdf(k)
        k2 = gauss_kuzmin.invcdf(p)
        assert mpmath.almosteq(k2, k)


def test_sf():
    one = mpmath.mp.one
    two = mpmath.mpf(2)
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.sf(k)
        assert mpmath.almosteq(p, mpmath.log((k + two)/(k + one), b=2))


def test_sf_invsf_roundtrip():
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.sf(k)
        k2 = gauss_kuzmin.invsf(p)
        assert mpmath.almosteq(k2, k)


def test_median():
    median = gauss_kuzmin.median()
    assert median == 2


def test_mode():
    mode = gauss_kuzmin.mode()
    assert mode == 1
