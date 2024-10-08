
import pytest
from mpmath import mp
from mpsci.distributions import gamma
from ._utils import call_and_check_mle
from ._expect import (
    check_entropy_with_integral,
    noncentral_moment_with_integral,
)


@mp.workdps(25)
def test_pdf_outside_support():
    p = gamma.pdf(-10, k=0.5, scale=1)
    assert p == 0


@mp.workdps(25)
def test_logpdf_outside_support():
    logp = gamma.logpdf(-10, k=0.5, scale=1)
    assert logp == mp.ninf


# This is the value of CDF[GammaDistribution[2, 3], 1] computed with
# Wolfram Alpha
_cdf123str = ("0.04462491923494766609919453743282711006251725357136112381217"
              "305716359456368679682785494521477976673845")


@mp.workdps(80)
def test_cdf():
    c = gamma.cdf(1, k=2, scale=3)
    expected = mp.mpf(_cdf123str)
    assert mp.almosteq(c, expected)


@mp.workdps(25)
def test_cdf_outside_support():
    c = gamma.cdf(-10, k=0.5, scale=1)
    assert c == 0


@mp.workdps(80)
def test_sf():
    s = gamma.sf(1, k=2, scale=3)
    expected = 1 - mp.mpf(_cdf123str)
    assert mp.almosteq(s, expected)


@mp.workdps(25)
def test_sf_outside_support():
    c = gamma.sf(-10, k=0.5, scale=1)
    assert c == 1


@pytest.mark.parametrize('x, k, scale', [(0.01, 2, 3),
                                         (1, 2, 3),
                                         (0.5, 1, 1),
                                         (10.0, 5, 1),
                                         (10.0, 0.015625, 1)])
@mp.workdps(50)
def test_cdf_invcdf_roundtrip(x, k, scale):
    p = gamma.cdf(x, k, scale)
    x1 = gamma.invcdf(p, k, scale)
    assert mp.almosteq(x1, x)


@pytest.mark.parametrize('x, k, scale', [(0.01, 2, 3),
                                         (1, 2, 3),
                                         (0.5, 1, 1),
                                         (10.0, 5, 1),
                                         (10.0, 0.015625, 1)])
@mp.workdps(50)
def test_sf_invsf_roundtrip(x, k, scale):
    p = gamma.sf(x, k, scale)
    x1 = gamma.invsf(p, k, scale)
    assert mp.almosteq(x1, x)


@mp.workdps(25)
def test_variance():
    k = 2.5
    scale = 3
    v = gamma.var(k=k, scale=scale)
    assert v == k*scale**2


@mp.workdps(25)
def test_skewness():
    s = gamma.skewness(k=16, scale=3)
    assert s == 0.5


@mp.workdps(25)
def test_kurtosis():
    k = gamma.kurtosis(k=16, scale=3)
    assert k == 3/8


@pytest.mark.parametrize('k', [0.25, 16])
@mp.workdps(25)
def test_entropy_with_integral(k):
    scale = 4.0
    check_entropy_with_integral(gamma, (k, scale), extradps=100)


@mp.workdps(80)
def test_interval_prob():
    x1 = 2.0
    x2 = 3.0
    p = gamma.interval_prob(x1, x2, 2, 1)
    expected = gamma.cdf(x2, 2, 1) - gamma.cdf(x1, 2, 1)
    assert mp.almosteq(p, expected)


@mp.workdps(80)
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
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(80)
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
    expected = mp.mpf(valstr)
    assert mp.almosteq(p, expected)


@mp.workdps(80)
def test_noncentral_moment():
    k = 2.5
    scale = 3.5

    m0 = gamma.noncentral_moment(0, k, scale)
    assert m0 == 1

    m1 = gamma.noncentral_moment(1, k, scale)
    assert m1 == gamma.mean(k, scale)

    m2 = gamma.noncentral_moment(2, k, scale)
    expected_m2 = noncentral_moment_with_integral(2, gamma, (k, scale))
    assert mp.almosteq(m2, expected_m2)

    m3 = gamma.noncentral_moment(3, k, scale)
    expected_m3 = noncentral_moment_with_integral(3, gamma, (k, scale))
    assert mp.almosteq(m3, expected_m3)


@mp.workdps(25)
def test_mom():
    x = [1, 2, 3, 4]
    k, scale = gamma.mom(x)
    assert k == mp.mpf(5)
    assert scale == mp.mpf('0.5')


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [63.0, 29.5, 53.0, 32.5, 22.0, 38.5, 24.5, 28.5, 23.5, 45.0, 64.5,
      13.0, 21.5, 40.0, 48.0, 22.0, 15.0, 47.5, 48.0, 26.5, 33.0, 21.0,
      28.0, 23.5, 16.0, 39.5, 38.5, 43.5, 23.0, 22.5, 33.5, 92.0, 44.5]]
)
@mp.workdps(50)
def test_mle(x):
    call_and_check_mle(gamma.mle, gamma.nll, x)


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [63.0, 29.5, 53.0, 32.5, 22.0, 38.5, 24.5, 28.5, 23.5, 45.0, 64.5,
      13.0, 21.5, 40.0, 48.0, 22.0, 15.0, 47.5, 48.0, 26.5, 33.0, 21.0,
      28.0, 23.5, 16.0, 39.5, 38.5, 43.5, 23.0, 22.5, 33.5, 92.0, 44.5]]
)
@mp.workdps(50)
def test_mle_fixed_scale(x):
    call_and_check_mle(lambda x: gamma.mle(x, scale=2)[0],
                       lambda x, k: gamma.nll(x, k, scale=2), x)


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [63.0, 29.5, 53.0, 32.5, 22.0, 38.5, 24.5, 28.5, 23.5, 45.0, 64.5,
      13.0, 21.5, 40.0, 48.0, 22.0, 15.0, 47.5, 48.0, 26.5, 33.0, 21.0,
      28.0, 23.5, 16.0, 39.5, 38.5, 43.5, 23.0, 22.5, 33.5, 92.0, 44.5]]
)
@mp.workdps(50)
def test_mle_fixed_k(x):
    call_and_check_mle(lambda x: gamma.mle(x, k=5)[1],
                       lambda x, scale: gamma.nll(x, k=5, scale=scale), x)


def test_nll_bad_x():
    with pytest.raises(ValueError, match='All values in x'):
        gamma.nll([1, 3, -1.5, 2], 2, 3)
