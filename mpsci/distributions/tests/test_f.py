
import pytest
from mpmath import mp
from mpsci.distributions import f


@mp.workdps(25)
def test_pdf():
    p = f.pdf(1.5, 10, 12)
    # The expected result was computed with Wolfram Alpha:
    #   PDF[FRatioDistribution[10, 12], 3/2]
    assert mp.almosteq(p, mp.mpf('3584000000/10460353203'))


def test_pdf_outside_support():
    p = f.pdf(-1.5, 10, 12)
    # Outside the support, so p must be 0.
    assert p == 0


@mp.workdps(25)
def test_logpdf():
    logp = f.logpdf(1.5, 10, 12)
    # The expected result was computed with Wolfram Alpha:
    #   PDF[FRatioDistribution[10, 12], 3/2]
    assert mp.almosteq(logp, mp.log(mp.mpf('3584000000/10460353203')))


def test_logpdf_outside_support():
    logp = f.logpdf(-1.5, 10, 12)
    # Outside the support, so logp must be -inf.
    assert logp == mp.ninf


@mp.workdps(25)
def test_cdf():
    p = f.cdf(1.5, 10, 12)
    # The expected result was computed with Wolfram Alpha:
    #   CDF[FRatioDistribution[10, 12], 3/2]
    assert mp.almosteq(p, mp.mpf('32290625/43046721'))


@mp.workdps(25)
def test_cdf_negative_x():
    p = f.cdf(-1.5, 10, 12)
    # x < 0, so p must be 0.
    assert p == 0


@mp.workdps(25)
def test_invcdf_roundtrip():
    dfn = 10
    dfd = 14
    p = 0.125
    x = f.invcdf(p, dfn, dfd)
    p2 = f.cdf(x, dfn, dfd)
    assert mp.almosteq(p2, p)


# Reference values were computed with Wolfram Alpha, e.g.
#    InverseCDF[FRatioDistribution[10, 12], 1/8]
@pytest.mark.parametrize(
    'p, dfn, dfd, ref',
    [('1/8', 10, 12,
      '0.47719175880508967810502455907077596240147311395460'),
     ('1/1073741824', 15, 23,
      '0.024300528764277007428004614052754091461622381277383859'),
     ('1048575/1048576', 4, 17,
      '23.5604958111857286837080521348245260943091665172894')]
)
@mp.workdps(40)
def test_invcdf_basic(p, dfn, dfd, ref):
    p = mp.mpf(p)
    ref = mp.mpf(ref)
    x = f.invcdf(p, dfn, dfd)
    assert mp.almosteq(x, ref)


@mp.workdps(25)
def test_sf():
    p = f.sf(1.5, 10, 12)
    # The expected result was computed with Wolfram Alpha:
    #   CDF[FRatioDistribution[10, 12], 3/2]
    assert mp.almosteq(p, 1 - mp.mpf('32290625/43046721'))


@mp.workdps(25)
def test_sf_negative_x():
    p = f.sf(-1.5, 10, 12)
    # x < 0, so p must be 1.
    assert p == 1


@mp.workdps(25)
def test_invsf_roundtrip():
    dfn = 10
    dfd = 14
    p = 0.125
    x = f.invsf(p, dfn, dfd)
    p2 = f.sf(x, dfn, dfd)
    assert mp.almosteq(p2, p)


# Reference values were computed with Wolfram Alpha, e.g.
#    InverseCDF[FRatioDistribution[10, 12], 1/8]
# See test_invcdf_basic above.  The p argument is converted
# to 1 - p so the same data from Wolfram can be used here.
@pytest.mark.parametrize(
    'p, dfn, dfd, ref',
    [('7/8', 10, 12,
      '0.47719175880508967810502455907077596240147311395460'),
     ('1073741823/1073741824', 15, 23,
      '0.024300528764277007428004614052754091461622381277383859'),
     ('1/1048576', 4, 17,
      '23.5604958111857286837080521348245260943091665172894')]
)
@mp.workdps(40)
def test_invsf_basic(p, dfn, dfd, ref):
    p = mp.mpf(p)
    ref = mp.mpf(ref)
    x = f.invsf(p, dfn, dfd)
    assert mp.almosteq(x, ref)


@mp.workdps(25)
def test_mean():
    m = f.mean(10, 12)
    # Value was double-checked with Wolfram Alpha:
    #   Mean[FRatioDistribution[10, 12]]
    assert mp.almosteq(m, mp.mpf('12/10'))


@mp.workdps(25)
def test_mean_small_dfd():
    m = f.mean(5, 1)
    # dfd <= 2, so the result must be inf.
    assert m == mp.inf


@mp.workdps(25)
def test_var():
    v = f.var(10, 12)
    # Value was double-checked with Wolfram Alpha:
    #   Variance[FRatioDistribution[10, 12]]
    assert mp.almosteq(v, mp.mpf('18/25'))


@mp.workdps(25)
def test_var_small_dfd():
    m = f.var(5, 1)
    # dfd <= 4, so the result must be inf.
    assert m == mp.inf


@mp.workdps(50)
def test_entropy_with_integral():
    dfn = 10
    dfd = 12
    entr = f.entropy(dfn, dfd)

    with mp.extradps(2*mp.dps):

        def integrand(t):
            return f.pdf(t, dfn, dfd) * f.logpdf(t, dfn, dfd)

        intgrl = -mp.quad(integrand, [0, mp.inf])

    assert mp.almosteq(entr, intgrl)
