
import pytest
import mpmath
from mpsci.distributions import beta


mpmath.mp.dps = 40


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
def test_pmf_integer_ab_half(a, b):
    half = mpmath.mpf('0.5')
    p = beta.pdf(half, a, b)
    assert p == mpmath.power(half, a + b - 2) / mpmath.beta(a, b)


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
def test_pmf_integer_ab_fourth(a, b):
    fourth = mpmath.mpf('0.25')
    p = beta.pdf(fourth, a, b)
    expected = (mpmath.power(3, b - 1) / mpmath.power(4, a + b - 2) /
                mpmath.beta(a, b))
    assert mpmath.almosteq(p, expected)


def test_interval_prob_close_x1_x2():
    save_dps = mpmath.mp.dps
    mpmath.mp.dps = 16
    try:
        x1 = 0.8
        x2 = 0.800000000000002
        p = beta.interval_prob(x1, x2, 0.5, 3.5)
        # fracions.Fraction(x2) is Fraction(1801439850948203, 2251799813685248)
        # The expression
        # `CDF[BetaDistribution[1/2, 7/2], 1801439850948203/2251799813685248]
        #  - CDF[BetaDistribution[1/2, 7/2], 8/10]`
        # on Wolfram Alpha gives:
        valstr = ("4.161579103212652227484856905835144318812644271729929336"
                  "1836438318375817555752008625301e-17")
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(p, expected)
    finally:
        mpmath.mp.dps = save_dps
