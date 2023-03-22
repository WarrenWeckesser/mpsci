
import pytest
from mpmath import mp
from mpsci.distributions import kumaraswamy


mp.dps = 40


#
# Expected values were computed with Wolfram Alpha, e.g.:
#
#     PDF[KumaraswamyDistribution[2, 3], 1/4]
#
# return 675/512.
#
@pytest.mark.parametrize(
    'x, a, b, expected',
    [(1/4, 2, 3, mp.mpf('675/512')),
     (1/4, 6, 0.5, 1/(mp.mpf(16)*mp.sqrt(455))),
     (5/8, 6, 0.5, 3125/(mp.mpf(448)*mp.sqrt(559)))],
)
def test_pdf_logpdf(x, a, b, expected):
    p = kumaraswamy.pdf(x, a, b)
    assert mp.almosteq(p, expected, rel_eps=10**(-mp.dps + 12))
    logp = kumaraswamy.logpdf(x, a, b)
    assert mp.almosteq(logp, mp.log(expected), rel_eps=10**(-mp.dps + 12))


#
# Expected values were computed with Wolfram Alpha, e.g.:
#
#     CDF[KumaraswamyDistribution[2, 3], 1/4]
#
# return 721/4096.
#
@pytest.mark.parametrize(
    'x, a, b, expected',
    [(1/4, 2, 3, mp.mpf('721/4096')),
     (1/4, 6, 0.5, 1 - 3*mp.sqrt(455)/64),
     (5/8, 6, 0.5, 1 - 21*mp.sqrt(559)/512)],
)
def test_cdf_sf(x, a, b, expected):
    c = kumaraswamy.cdf(x, a, b)
    assert mp.almosteq(c, expected, rel_eps=10**(-mp.dps + 12))
    s = kumaraswamy.sf(x, a, b)
    assert mp.almosteq(s, 1 - expected, rel_eps=10**(-mp.dps + 12))


def test_cdf_invcdf_roundtrip():
    x = mp.mpf(0.75)
    p = kumaraswamy.cdf(x, 0.5, 7.5)
    x2 = kumaraswamy.invcdf(p, 0.5, 7.5)
    assert mp.almosteq(x, x2)


def test_sf_invsf_roundtrip():
    x = mp.mpf(0.75)
    p = kumaraswamy.sf(x, 0.5, 7.5)
    x2 = kumaraswamy.invsf(p, 0.5, 7.5)
    assert mp.almosteq(x, x2)


def test_mean():
    m = kumaraswamy.mean(6, 0.5)
    # Expected value was computed with Wolfram Alpha:
    #    Mean[KumaraswamyDistribution[6, 1/2]]
    expected = mp.mpf('0.9107439929578431044324781336678492182081380668')
    assert mp.almosteq(m, expected, rel_eps=10**(-mp.dps + 12))


def test_var():
    v = kumaraswamy.var(6, 0.5)
    # Expected value was computed with Wolfram Alpha:
    #    Variance[KumaraswamyDistribution[6, 1/2]]
    expected = mp.mpf('0.0118546424864767865065319749025960710717792871')
    assert mp.almosteq(v, expected, rel_eps=10**(-mp.dps + 12))


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
