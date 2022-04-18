
import mpmath
from mpsci.distributions import gamma


mpmath.mp.dps = 16


# This is the value of CDF[GammaDistribution[2, 3], 1] computed with
# Wolfram Alpha
_cdf123str = ("0.04462491923494766609919453743282711006251725357136112381217"
              "305716359456368679682785494521477976673845")


def test_cdf():
    c = gamma.cdf(1, k=2, theta=3)
    expected = mpmath.mpf(_cdf123str)
    assert mpmath.almosteq(c, expected)


def test_sf():
    s = gamma.sf(1, k=2, theta=3)
    expected = 1 - mpmath.mpf(_cdf123str)
    assert mpmath.almosteq(s, expected)


def test_skewness():
    s = gamma.skewness(k=16, theta=3)
    assert s == 0.5


def test_kurtosis():
    k = gamma.kurtosis(k=16, theta=3)
    assert k == 3/8


def test_interval_prob():
    x1 = 2.0
    x2 = 3.0
    p = gamma.interval_prob(x1, x2, 2, 1)
    expected = gamma.cdf(x2, 2, 1) - gamma.cdf(x1, 2, 1)
    assert mpmath.almosteq(p, expected)


def test_interval_prob_x1_x2_close():
    # With mpmath.mps.dps = 16, this test would fail if interval_prob
    # was computed as the difference of the CDF values.
    x1 = 2.0
    x2 = 2.000000000000002
    p = gamma.interval_prob(x1, x2, 2, 1)
    # fractions.Fraction(x2) is Fraction(4503599627370501, 2251799813685248).
    # On Wolfram Alpha, the expression
    # `CDF[GammaDistribution[2, 1], 4503599627370501/2251799813685248]
    #  - CDF[GammaDistribution[2, 1], 2]`
    # gives the following:
    valstr = ("6.0100938997381721747992665891975556663629544173547573"
              "3648532458638642871160406995189e-16")
    expected = mpmath.mpf(valstr)
    assert mpmath.almosteq(p, expected)


def test_interval_prob_close_cdf_values():
    # With mpmath.mps.dps = 16, this test would fail if interval_prob
    # was computed as the difference of the CDF values.
    x1 = 44
    x2 = 45
    p = gamma.interval_prob(x1, x2, 2, 1)
    # On Wolfram Alpha, the expression
    # `CDF[GammaDistribution[2, 1], 45] - CDF[GammaDistribution[2, 1], 44]`
    # gives the following:
    valstr = ("2.18475096145748735561473657875134307117716635897865537"
              "4359464779918152310943256631294315558843429839e-18")
    expected = mpmath.mpf(valstr)
    assert mpmath.almosteq(p, expected)


def test_mom():
    x = [1, 2, 3, 4]
    k, theta = gamma.mom(x)
    assert k == mpmath.mpf(5)
    assert theta == mpmath.mpf('0.5')
