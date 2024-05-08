from itertools import product
import pytest
from mpmath import mp
from mpsci.stats import mean, var
from mpsci.distributions import beta
from ._expect import check_entropy_with_integral


@mp.workdps(25)
def test_pdf_out_of_support():
    x = beta.pdf(-1, 2, 3)
    assert x == 0
    x = beta.pdf(10, 4, 2)
    assert x == 0


@mp.workdps(25)
def test_pdf_edge_cases():
    x = beta.pdf(0, 0.25, 3)
    assert x == mp.inf
    x = beta.pdf(1, 5, 0.125)
    assert x == mp.inf


@mp.workdps(25)
def test_logpdf_out_of_support():
    x = beta.logpdf(-1, 2, 3)
    assert x == mp.ninf
    x = beta.logpdf(10, 4, 2)
    assert x == mp.ninf


@mp.workdps(25)
def test_cdf_out_of_support():
    x = beta.cdf(-3, 0.5, 2)
    assert x == 0
    x = beta.cdf(2.5, 1, 4)
    assert x == 1


@mp.workdps(25)
def test_sf_out_of_support():
    x = beta.sf(-3, 0.5, 2)
    assert x == 1
    x = beta.sf(2.5, 1, 4)
    assert x == 0


@mp.workdps(50)
def test_invcdf():
    p = mp.mpf('0.1')
    a = 2
    b = 3
    x = beta.invcdf(p, a, b)
    # Reference value was computed with Wolfram Alpha:
    #     InverseCDF[BetaDistribution[2, 3], 1/10]
    xref = mp.mpf('0.142559316710030719125115018772962783084464477318837')
    # '4777035373472136044995919668643934344200183664243')
    assert mp.almosteq(x, xref)


@mp.workdps(50)
def test_invsf():
    p = mp.mpf('0.1')
    a = 2
    b = 3
    x = beta.invsf(p, a, b)
    # Reference value was computed with Wolfram Alpha:
    #     InverseCDF[BetaDistribution[2, 3], 9/10]
    xref = mp.mpf('0.679539416278181674859785973927344723631083369808448')
    assert mp.almosteq(x, xref)


@mp.workdps(50)
def test_invcdf_cdf_roundtrip():
    p0 = mp.mpf('0.6')
    a = mp.mpf('0.95')
    b = mp.mpf(2)
    x = beta.invcdf(p0, a, b)
    p1 = beta.cdf(x, a, b)
    assert mp.almosteq(p1, p0)


@mp.workdps(50)
def test_invsf_sf_roundtrip():
    p0 = mp.mpf('0.6')
    a = mp.mpf('0.95')
    b = mp.mpf(2)
    x = beta.invsf(p0, a, b)
    p1 = beta.sf(x, a, b)
    assert mp.almosteq(p1, p0)


@mp.workdps(50)
def test_mean():
    a = mp.mpf('0.75')
    b = mp.mpf('4.5')
    m = beta.mean(a, b)
    expected = a / (a + b)  # See, e.g. wikipedia, or any other ref.
    assert mp.almosteq(m, expected)


@pytest.mark.parametrize('a, b, expected',
                         [(1, 3, '0.0375'),
                          (0.5, 0.5, 0.125)])
@mp.workdps(50)
def test_var(a, b, expected):
    expected = mp.mpf(expected)
    v = beta.var(a, b)
    assert mp.almosteq(v, expected)


def test_skewness():
    a = 1
    b = 3
    # Expected value computed with Wolfram Alpha:
    #     Skewness[BetaDistribution[1, 3]]
    s = '0.860662965823870418928725644396088802407315934509242405908'
    with mp.workdps(len(s) - 2):
        v = beta.skewness(a, b)
        expected = mp.mpf(s)
        assert mp.almosteq(v, expected)


@mp.workdps(50)
def test_kurtosis():
    a = 1
    b = 3
    kurt = beta.kurtosis(a, b)
    expected = mp.mpf(2.0) / 21
    assert mp.almosteq(kurt, expected)


@mp.workdps(50)
def test_noncentral_moment():
    a = 1.5
    b = 3.0

    m0 = beta.noncentral_moment(0, a, b)
    assert m0 == 1

    m1 = beta.noncentral_moment(1, a, b)
    assert m1 == beta.mean(a, b)

    m2 = beta.noncentral_moment(2, a, b)
    assert mp.almosteq(m2, beta.var(a, b) + beta.mean(a, b)**2)


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
@mp.workdps(50)
def test_pdf_integer_ab_half(a, b):
    half = mp.mpf('0.5')
    p = beta.pdf(half, a, b)
    expected = mp.power(half, a + b - 2) / mp.beta(a, b)
    assert p == expected
    logp = beta.logpdf(half, a, b)
    assert mp.almosteq(logp, mp.log(expected))


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
@mp.workdps(50)
def test_pdf_integer_ab_fourth(a, b):
    fourth = mp.mpf('0.25')
    p = beta.pdf(fourth, a, b)
    expected = (mp.power(3, b - 1) / mp.power(4, a + b - 2) /
                mp.beta(a, b))
    assert mp.almosteq(p, expected)
    logp = beta.logpdf(fourth, a, b)
    assert mp.almosteq(logp, mp.log(expected))


@mp.workdps(25)
def test_interval_prob_x_validation():
    with pytest.raises(ValueError, match='x1 must not be greater than x2'):
        beta.interval_prob(0.5, 0.25, 3.5, 8)


def test_interval_prob_close_x1_x2():
    with mp.workdps(16):
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
        expected = mp.mpf(valstr)
        assert mp.almosteq(p, expected)


@mp.workdps(50)
def test_entropy_with_integral():
    check_entropy_with_integral(beta, (0.75, 2.75))


@mp.workdps(25)
def test_mle_trivial_case():
    x = [0.25, 0.5, 0.625, 0.875]
    a = 1.5
    b = 4.0
    # Trivial case: both parameters fixed.
    ahat, bhat = beta.mle(x, a=a, b=b)
    assert (ahat, bhat) == (a, b)


@mp.workdps(50)
def test_mle():
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mle(x)

    N = len(x)

    ea = mp.fsum([mp.log1p(-t) for t in x])/N
    # First order condition for the MLE.
    ca = ea + mp.psi(0, ahat + bhat) - mp.psi(0, bhat)
    assert mp.almosteq(ca, 0)

    eb = mp.fsum([mp.log(t) for t in x])/N
    # First order condition for the MLE.
    cb = eb + mp.psi(0, ahat + bhat) - mp.psi(0, ahat)
    assert mp.almosteq(cb, 0)


@pytest.mark.parametrize(
    'x',
    [[0.01, 0.05, 0.125, 0.375],
     [0.43, 0.51, 0.625, 0.75, 0.875, 0.9925]]
)
def test_mle_minimizes_nll(x):
    with mp.workdps(40):
        a_hat, b_hat = beta.mle(x)
        nll = beta.nll(x, a=a_hat, b=b_hat)
        delta = 1e-9
        n = 2
        dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
        for d in dirs:
            a = a_hat + d[0]*delta
            b = b_hat + d[1]*delta
            assert nll < beta.nll(x, a=a, b=b)


@mp.workdps(50)
def test_mle_b_fixed():
    b = mp.mpf('1.25')
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mle(x, b=b)
    assert bhat == b  # because b was fixed.

    N = len(x)
    ea = mp.fsum([mp.log(t) for t in x])/N
    # First order condition for the MLE.
    ca = ea + mp.psi(0, ahat + bhat) - mp.psi(0, ahat)
    assert mp.almosteq(ca, 0)


@mp.workdps(50)
def test_mle_a_fixed():
    a = mp.mpf('2.75')
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mle(x, a=a)
    assert ahat == a  # because a was fixed.

    N = len(x)
    eb = mp.fsum([mp.log1p(-t) for t in x])/N
    # First order condition for the MLE.
    cb = eb + mp.psi(0, ahat + bhat) - mp.psi(0, bhat)
    assert mp.almosteq(cb, 0)


@mp.workdps(50)
def test_mom():
    # https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters
    x = [0.25, 0.5, 0.625, 0.875]
    ahat, bhat = beta.mom(x)

    xbar = mean(x)
    vbar = var(x)
    p = xbar*(1 - xbar)/vbar - 1
    assert mp.almosteq(ahat, xbar*p)
    assert mp.almosteq(bhat, (1 - xbar)*p)


@mp.workdps(50)
def test_bad_a_b():
    with pytest.raises(ValueError, match='must be greater than 0'):
        beta.pdf(0.25, -1.0, 2.5)
