
import pytest
import mpmath
from mpsci.stats import mean, var
from mpsci.distributions import beta


mpmath.mp.dps = 50


def test_invcdf_cdf_roundtrip():
    p0 = mpmath.mpf('0.6')
    a = mpmath.mpf('0.95')
    b = mpmath.mpf(2)
    x = beta.invcdf(p0, a, b)
    p1 = beta.cdf(x, a, b)
    assert mpmath.almosteq(p1, p0)


def test_invsf_sf_roundtrip():
    p0 = mpmath.mpf('0.6')
    a = mpmath.mpf('0.95')
    b = mpmath.mpf(2)
    x = beta.invsf(p0, a, b)
    p1 = beta.sf(x, a, b)
    assert mpmath.almosteq(p1, p0)


def test_mean():
    a = mpmath.mpf('0.75')
    b = mpmath.mpf('4.5')
    m = beta.mean(a, b)
    expected = a / (a + b)  # See, e.g. wikipedia, or any other ref.
    assert mpmath.almosteq(m, expected)


@pytest.mark.parametrize('a, b, expected',
                         [(1, 3, mpmath.mpf('0.0375')),
                          (0.5, 0.5, 0.125)])
def test_var(a, b, expected):
    with mpmath.workdps(25):
        v = beta.var(a, b)
        assert mpmath.almosteq(v, expected)


def test_skewness():
    a = 1
    b = 3
    # Expected value computed with Wolfram Alpha:
    #     Skewness[BetaDistribution[1, 3]]
    s = '0.860662965823870418928725644396088802407315934509242405908'
    with mpmath.workdps(len(s) - 2):
        v = beta.skewness(a, b)
        expected = mpmath.mpf(s)
        assert mpmath.almosteq(v, expected)


def test_kurtosis():
    a = 1
    b = 3
    with mpmath.workdps(25):
        kurt = beta.kurtosis(a, b)
        expected = mpmath.mpf(2.0) / 21
        assert mpmath.almosteq(kurt, expected)


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
def test_pdf_integer_ab_half(a, b):
    half = mpmath.mpf('0.5')
    p = beta.pdf(half, a, b)
    expected = mpmath.power(half, a + b - 2) / mpmath.beta(a, b)
    assert p == expected
    logp = beta.logpdf(half, a, b)
    assert mpmath.almosteq(logp, mpmath.log(expected))


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
def test_pdf_integer_ab_fourth(a, b):
    fourth = mpmath.mpf('0.25')
    p = beta.pdf(fourth, a, b)
    expected = (mpmath.power(3, b - 1) / mpmath.power(4, a + b - 2) /
                mpmath.beta(a, b))
    assert mpmath.almosteq(p, expected)
    logp = beta.logpdf(fourth, a, b)
    assert mpmath.almosteq(logp, mpmath.log(expected))


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


def test_mle():
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mle(x)

    N = len(x)

    ea = mpmath.fsum([mpmath.log1p(-t) for t in x])/N
    # First order condition for the MLE.
    ca = ea + mpmath.psi(0, ahat + bhat) - mpmath.psi(0, bhat)
    assert mpmath.almosteq(ca, 0)

    eb = mpmath.fsum([mpmath.log(t) for t in x])/N
    # First order condition for the MLE.
    cb = eb + mpmath.psi(0, ahat + bhat) - mpmath.psi(0, ahat)
    assert mpmath.almosteq(cb, 0)


def test_mle_b_fixed():
    b = mpmath.mpf('1.25')
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mle(x, b=b)
    assert bhat == b  # because b was fixed.

    N = len(x)
    ea = mpmath.fsum([mpmath.log(t) for t in x])/N
    # First order condition for the MLE.
    ca = ea + mpmath.psi(0, ahat + bhat) - mpmath.psi(0, ahat)
    assert mpmath.almosteq(ca, 0)


def test_mle_a_fixed():
    a = mpmath.mpf('2.75')
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mle(x, a=a)
    assert ahat == a  # because a was fixed.

    N = len(x)
    eb = mpmath.fsum([mpmath.log1p(-t) for t in x])/N
    # First order condition for the MLE.
    cb = eb + mpmath.psi(0, ahat + bhat) - mpmath.psi(0, bhat)
    assert mpmath.almosteq(cb, 0)


def test_mom():
    # https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mom(x)

    xbar = mean(x)
    vbar = var(x)
    p = xbar*(1 - xbar)/vbar - 1
    assert mpmath.almosteq(ahat, xbar*p)
    assert mpmath.almosteq(bhat, (1 - xbar)*p)


def test_bad_a_b():
    with pytest.raises(ValueError, match='must be greater than 0'):
        beta.pdf(0.25, -1.0, 2.5)
