import pytest
from mpmath import mp
from mpsci.distributions import benini
from ._expect import noncentral_moment_with_integral


@pytest.mark.parametrize(
    'x, a, b, scale, refstr',
    [(7.5, 3, 4, 5, '0.12779474416889056900129594315350610580454836'),
     (100, 16, 5, 10, '1.197917519051586714244438133806848647531374908e-28'),
     (0.5, 4, 5, 0.5, '8')]
)
@mp.workdps(40)
def test_pdf(x, a, b, scale, refstr):
    p = benini.pdf(x, a, b, scale)
    assert mp.almosteq(p, mp.mpf(refstr))


@mp.workdps(50)
def test_cdf():
    x = 4.5
    a = 2.0
    b = 3.75
    scale = 3.0
    p = benini.cdf(x, a, b, scale)
    # From Wolfram Alpha
    #     CDF[BeniniDistribution[2, 15/4, 3], 9/2]
    refstr = '0.760077072664075321111764352700581449053704882746012742865'
    assert mp.almosteq(p, mp.mpf(refstr))


@mp.workdps(50)
def test_logcdf():
    x = 45
    a = 3
    b = 7
    scale = 2
    p = benini.logcdf(x, a, b, scale)
    # From Wolfram Alpha
    #   N[Log[CDF[BeniniDistribution[3, 7, 2], 45]], 51]
    refstr = '-2.97279970187999773599940731080551000328862307387356e-34'
    assert mp.almosteq(p, mp.mpf(refstr))


@mp.workdps(50)
def test_cdf_invcdf_round_trip():
    x = 13
    a = 1
    b = 2
    scale = 3
    p = benini.cdf(x, a, b, scale)
    x2 = benini.invcdf(p, a, b, scale)
    assert mp.almosteq(x2, x)


@mp.workdps(50)
def test_sf():
    x = 45
    a = 3
    b = 10
    scale = 3.0
    p = benini.sf(x, a, b, scale)
    # From Wolfram Alpha
    #   N[SurvivalFunction[BeniniDistribution[3, 10, 3], 45], 51]
    refstr = '4.19357598749336544648688110723440000222445241768311e-36'
    assert mp.almosteq(p, mp.mpf(refstr))


@mp.workdps(50)
def test_logsf():
    x = 25
    a = 21
    b = 1
    scale = 5
    p = benini.logsf(x, a, b, scale)
    # From Wolfram Alpha
    #   N[Log[SurvivalFunction[BeniniDistribution[21, 1, 5], 25]], 51]
    refstr = '-36.3884865550963428117961170027425953254300826062383'
    assert mp.almosteq(p, mp.mpf(refstr))


@mp.workdps(50)
def test_sf_invsf_round_trip():
    x = 13
    a = 3
    b = 1
    scale = 2
    p = benini.sf(x, a, b, scale)
    x2 = benini.invsf(p, a, b, scale)
    assert mp.almosteq(x2, x)


@pytest.mark.parametrize('b', [1, 6])
def test_mode_endpoint(b):
    a = 3
    scale = 2.5
    m = benini.mode(a, b, scale)
    assert m == scale


@mp.workdps(50)
def test_mode_interior():
    # A crude test of the mode.
    a = 10
    b = 60  # b > a*(a + 1)/2
    scale = 1
    m = benini.mode(a, b, scale)
    pm = benini.pdf(m, a, b, scale)
    delta = mp.sqrt(mp.eps)
    assert benini.pdf(m - delta, a, b, scale) < pm
    assert benini.pdf(m + delta, a, b, scale) < pm


@pytest.mark.parametrize('alpha, beta, scale',
                         [(0.125, 25, 1),
                          (0.125, 25, 3),
                          (10, 0.25, 0.25),
                          (100, 80, 1)])
@mp.workdps(50)
def test_mean_with_integral(alpha, beta, scale):
    m = benini.mean(alpha, beta, scale)
    q = noncentral_moment_with_integral(1, benini, (alpha, beta, scale))
    assert mp.almosteq(m, q)


@mp.workdps(50)
def test_mean_against_wolfram_alpha():
    m = benini.mean(3, 4, 1)
    # From Wolfram Alpha:
    #    Mean[BeniniDistribution[3, 4, 1]]
    refstr = '1.272820680382523521049693913688605974058687129016880'
    assert mp.almosteq(m, mp.mpf(refstr))


@mp.workdps(50)
def test_var_with_integral():
    alpha = 0.125
    beta = 3.0
    scale = 2.0
    mu = benini.mean(alpha, beta, scale)
    var = benini.var(alpha, beta, scale)
    expected = mp.quad(lambda t: (t - mu)**2*benini.pdf(t, alpha, beta, scale),
                       [scale, mp.inf])
    assert mp.almosteq(var, expected)


@mp.workdps(50)
def test_var_against_wolfram_alpha():
    v = benini.var(3, 4, 5)
    # From Wolfram Alpha:
    #    Variance[BeniniDistribution[3, 4, 5]]
    refstr = '1.565734202983936195380092260357367181004500764935885673850'
    assert mp.almosteq(v, mp.mpf(refstr))
