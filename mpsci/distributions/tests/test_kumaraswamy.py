from itertools import product
import pytest
from mpmath import mp
from mpsci.distributions import kumaraswamy, Initial


#
# Expected values were computed with Wolfram Alpha, e.g.:
#
#     PDF[KumaraswamyDistribution[2, 3], 1/4]
#
# return 675/512.
#
@pytest.mark.parametrize(
    'x, a, b, expected',
    [(1/4, 2, 3, "mp.mpf('675/512')"),
     (1/4, 6, 0.5, "1/(mp.mpf(16)*mp.sqrt(455))"),
     (5/8, 6, 0.5, "3125/(mp.mpf(448)*mp.sqrt(559))")],
)
@mp.workdps(50)
def test_pdf_logpdf(x, a, b, expected):
    p = kumaraswamy.pdf(x, a, b)
    expected = eval(expected)
    assert mp.almosteq(p, expected, rel_eps=10**(-mp.dps + 12))
    logp = kumaraswamy.logpdf(x, a, b)
    assert mp.almosteq(logp, mp.log(expected), rel_eps=10**(-mp.dps + 12))


@pytest.mark.parametrize('a, b', [(-1, 3), (2.5, -2)])
def test_pdf_validation(a, b):
    with pytest.raises(ValueError, match='must be greater than 0'):
        kumaraswamy.pdf(0.75, a, b)


@pytest.mark.parametrize('x', [-0.5, 1.75])
def test_pdf_outside_support(x):
    p = kumaraswamy.pdf(x, 8.5, 1.25)
    assert p == 0


@pytest.mark.parametrize('x', [-0.5, 1.75])
def test_logpdf_outside_support(x):
    p = kumaraswamy.logpdf(x, 8.5, 1.25)
    assert p == mp.ninf


@pytest.mark.parametrize('x, a, b', [(0, 0.75, 2.5), (1, 3.5, 0.25)])
def test_pdf_special_cases(x, a, b):
    p = kumaraswamy.pdf(x, a, b)
    assert p == mp.inf


@pytest.mark.parametrize('x', [-0.5, 1.75])
def test_cdf_outside_support(x):
    p = kumaraswamy.cdf(x, 8.5, 1.25)
    expected = 0 if x < 0 else 1
    assert p == expected


@pytest.mark.parametrize('x', [-0.5, 1.75])
def test_sf_outside_support(x):
    p = kumaraswamy.sf(x, 8.5, 1.25)
    expected = 1 if x < 0 else 0
    assert p == expected


#
# Expected values were computed with Wolfram Alpha, e.g.:
#
#     CDF[KumaraswamyDistribution[2, 3], 1/4]
#
# return 721/4096.
#
@pytest.mark.parametrize(
    'x, a, b, expected',
    [(1/4, 2, 3, "mp.mpf('721/4096')"),
     (1/4, 6, 0.5, "1 - 3*mp.sqrt(455)/64"),
     (5/8, 6, 0.5, "1 - 21*mp.sqrt(559)/512")],
)
@mp.workdps(50)
def test_cdf_sf(x, a, b, expected):
    c = kumaraswamy.cdf(x, a, b)
    expected = eval(expected)
    assert mp.almosteq(c, expected, rel_eps=10**(-mp.dps + 12))
    s = kumaraswamy.sf(x, a, b)
    assert mp.almosteq(s, 1 - expected, rel_eps=10**(-mp.dps + 12))


@mp.workdps(50)
def test_cdf_invcdf_roundtrip():
    x = mp.mpf(0.75)
    p = kumaraswamy.cdf(x, 0.5, 7.5)
    x2 = kumaraswamy.invcdf(p, 0.5, 7.5)
    assert mp.almosteq(x, x2)


@mp.workdps(50)
def test_sf_invsf_roundtrip():
    x = mp.mpf(0.75)
    p = kumaraswamy.sf(x, 0.5, 7.5)
    x2 = kumaraswamy.invsf(p, 0.5, 7.5)
    assert mp.almosteq(x, x2)


def test_invcdf_edge_cases():
    p = kumaraswamy.invcdf(0, 1.5, 2.5)
    assert p == 0
    p = kumaraswamy.invcdf(1, 1.5, 2.5)
    assert p == 1


def test_invsf_edge_cases():
    p = kumaraswamy.invsf(0, 1.5, 2.5)
    assert p == 1
    p = kumaraswamy.invsf(1, 1.5, 2.5)
    assert p == 0


def test_median():
    a = 1.5
    b = 3.75
    m = kumaraswamy.median(a, b)
    p = kumaraswamy.cdf(m, a, b)
    assert mp.almosteq(p, 0.5)


@mp.workdps(50)
def test_mean():
    m = kumaraswamy.mean(6, 0.5)
    # Expected value was computed with Wolfram Alpha:
    #    Mean[KumaraswamyDistribution[6, 1/2]]
    expected = mp.mpf('0.9107439929578431044324781336678492182081380668')
    assert mp.almosteq(m, expected, rel_eps=10**(-mp.dps + 12))


@mp.workdps(50)
def test_var():
    v = kumaraswamy.var(6, 0.5)
    # Expected value was computed with Wolfram Alpha:
    #    Variance[KumaraswamyDistribution[6, 1/2]]
    expected = mp.mpf('0.0118546424864767865065319749025960710717792871')
    assert mp.almosteq(v, expected, rel_eps=10**(-mp.dps + 12))


@pytest.mark.parametrize('order', [0, 1, 2, 3, 4])
@mp.workdps(50)
def test_noncentral_moment_with_integral(order):
    a = 4.5
    b = 1.75
    m = kumaraswamy.noncentral_moment(order, a, b)
    intgrl = mp.quad(lambda t: t**order*kumaraswamy.pdf(t, a, b), [0, 1])
    assert mp.almosteq(m, intgrl)


@mp.workdps(50)
def test_skewness_with_integral():
    a = 1.125
    b = 8.5
    sk = kumaraswamy.skewness(a, b)

    # Skewness is E(((x - mu)/sigma)**3); compute the expected
    # value with an integral.
    mu = kumaraswamy.mean(a, b)
    sigma = mp.sqrt(kumaraswamy.var(a, b))
    intgrl = mp.quad(lambda t: kumaraswamy.pdf(t, a, b)*((t - mu)/sigma)**3,
                     [0, 1])

    assert mp.almosteq(sk, intgrl)


@mp.workdps(50)
def test_entropy_with_integral():
    a = 0.75
    b = 2.75
    entr = kumaraswamy.entropy(a, b)

    with mp.extradps(2*mp.dps):

        def integrand(t):
            return kumaraswamy.pdf(t, a, b) * kumaraswamy.logpdf(t, a, b)

        intgrl = -mp.quad(integrand, [0, 1])

    assert mp.almosteq(entr, intgrl)


@pytest.mark.parametrize(
    'x',
    [[0.01, 0.05, 0.125, 0.375],
     [0.43, 0.51, 0.625, 0.75, 0.875, 0.9925]]
)
@pytest.mark.parametrize('a', [None, Initial(1.0625)])
def test_mle(x, a):
    with mp.workdps(40):
        a_hat, b_hat = kumaraswamy.mle(x, a=a)
        nll = kumaraswamy.nll(x, a=a_hat, b=b_hat)
        delta = 1e-9
        n = 2
        dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
        for d in dirs:
            a = a_hat + d[0]*delta
            b = b_hat + d[1]*delta
            assert nll < kumaraswamy.nll(x, a=a, b=b)


@pytest.mark.parametrize(
    'x',
    [[0.01, 0.05, 0.125, 0.375],
     [0.43, 0.51, 0.625, 0.75, 0.875, 0.9925]]
)
def test_mle_fixed_a(x):
    with mp.workdps(40):
        a = 1.25
        a_hat, b_hat = kumaraswamy.mle(x, a=a)
        assert a_hat == a
        nll = kumaraswamy.nll(x, a=a, b=b_hat)
        delta = 1e-9
        assert nll < kumaraswamy.nll(x, a=a, b=b_hat + delta)
        assert nll < kumaraswamy.nll(x, a=a, b=b_hat - delta)


@pytest.mark.parametrize(
    'x',
    [[0.01, 0.05, 0.125, 0.375],
     [0.43, 0.51, 0.625, 0.75, 0.875, 0.9925]]
)
@pytest.mark.parametrize('a', [None, Initial(1.0625)])
def test_mle_fixed_b(x, a):
    with mp.workdps(40):
        b = 3
        a_hat, b_hat = kumaraswamy.mle(x, a=a, b=b)
        assert b_hat == b
        nll = kumaraswamy.nll(x, a=a_hat, b=b)
        delta = 1e-9
        assert nll < kumaraswamy.nll(x, a=a_hat + delta, b=b)
        assert nll < kumaraswamy.nll(x, a=a_hat - delta, b=b)


def test_mle_fixed_a_and_b():
    # Trivial case
    a = 1.25
    b = 3.0
    a_hat, b_hat = kumaraswamy.mle([0.01, 0.05, 0.125, 0.375], a=a, b=b)
    assert (a_hat, b_hat) == (a, b)
