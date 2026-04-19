import pytest
from mpmath import mp
from mpsci.distributions import laplace
from ._expect import noncentral_moment_with_integral


@mp.workdps(90)
def test_pdf():
    x = -100
    p1 = laplace.pdf(x)
    # Wolfram Alpha:
    #     PDF[LaplaceDistribution[0, 1],  -100]
    refstr = ('1.86003798801041798147984790193155916867944614618839098356030'
              '6938331645237947907859078559389321140748e-44')
    assert mp.almosteq(p1, mp.mpf(refstr))
    p2 = laplace.pdf(-x)
    assert mp.almosteq(p2, mp.mpf(refstr))


@mp.workdps(90)
def test_logpdf():
    x = 39
    logp1 = laplace.logpdf(x)
    # Wolfram Alpha:
    #     Log[PDF[LaplaceDistribution[0, 1], 39]]
    refstr = ('-39.69314718055994530941723212145817656807550013436025525412'
              '0680009493393621969694715605863326996418688')
    assert mp.almosteq(logp1, mp.mpf(refstr))
    logp2 = laplace.logpdf(-x)
    assert mp.almosteq(logp1, mp.mpf(refstr))


@mp.workdps(90)
def test_cdf():
    x = 125
    p1 = laplace.cdf(x)
    # Wolfram Alpha:
    #     CDF[LaplaceDistribution[0, 1], 125]
    refstr = ('0.9999999999999999999999999999999999999999999999999999997416'
              '78968358106950987364069632122106786209953')
    assert mp.almosteq(p1, mp.mpf(refstr))
    p2 = laplace.cdf(-x)
    assert mp.almosteq(p2, mp.one - mp.mpf(refstr))


@pytest.mark.parametrize('x', [-6, -8.5])
@mp.workdps(50)
def test_cdf_invcdf_round_trip(x):
    loc = -8
    scale = 0.125
    p = laplace.cdf(x, loc=loc, scale=scale)
    x2 = laplace.invcdf(p, loc=loc, scale=scale)
    assert mp.almosteq(x2, x)


@mp.workdps(90)
def test_sf():
    x = 125
    p1 = laplace.sf(x)
    # Wolfram Alpha:
    #     SurvivalFunction[LaplaceDistribution[0, 1], 125]
    refstr = ('2.5832103164189304901263593036787789321379004724067154776014'
              '31651177057083104023220463576736247307508e-55')
    assert mp.almosteq(p1, mp.mpf(refstr))
    p2 = laplace.sf(-x)
    assert mp.almosteq(p2, mp.one - mp.mpf(refstr))


@pytest.mark.parametrize('x', [13, -28.25])
@mp.workdps(50)
def test_sf_invsf_round_trip(x):
    loc = 3
    scale = 4.5
    p = laplace.sf(x, loc=loc, scale=scale)
    x2 = laplace.invsf(p, loc=loc, scale=scale)
    assert mp.almosteq(x2, x)


@mp.workdps(40)
def test_mle():
    x = [1, 2, 4, 4, 6, 8, 8, 9]
    loc_hat, scale_hat = laplace.mle(x)
    # The MLE for loc is the median.
    assert loc_hat == 5
    # The MLE for scale is the mean absolute deviation from the median:
    # scale_hat = mean(|1-5|, |2-5|, |4-5|, |4-5|, |6-5|,
    #                  |8-5|, |8-5|, |9-5|)
    #           = mean(4, 3, 1, 1, 1, 3, 3, 4)
    #           = 20/8 = 2.5
    assert scale_hat == 2.5


@mp.workdps(50)
def test_interval_prob():
    x1 = -1001
    x2 = -1000
    p = laplace.interval_prob(x1, x2)
    # s computed with Wolfram Alpha:
    #   CDF[LaplaceDistribution[0, 1], -1000] - CDF[LaplaceDistribution[0, 1], -1001]
    s = "1.6043089874548760814516280731829919474907824618811554928501e-435"
    ref = mp.mpf(s)
    assert mp.almosteq(p, ref)

    # Reuse above result, by symmetry.
    p = laplace.interval_prob(-x2, -x1)
    assert mp.almosteq(p, ref)

    x1 = -mp.mpf(1)/10000000
    x2 = mp.mpf(2)/10000000
    p = laplace.interval_prob(x1, x2)
    # Wolfram Alpha:
    #    CDF[LaplaceDistribution[0, 1], 2/10000000]
    #      - CDF[LaplaceDistribution[0, 1], -1/10000000]
    s = "1.4999998750000074999996458333470833328819444572420631733631023e-7"
    ref = mp.mpf(s)
    assert mp.almosteq(p, ref)


def test_interval_prob_bad_interval():
    with pytest.raises(ValueError, match="x1 must not be greater than x2"):
        laplace.interval_prob(2, 1, loc=3, scale=4)


def test_interval_prob_trivial():
    p = laplace.interval_prob(2.0, 2.0, loc=0.5, scale=0.25)
    assert p == 0


@pytest.mark.parametrize('loc, scale',
                         [(0.5, 3.0), (-10, 4), (125, 87.5)])
@mp.workdps(50)
def test_mean_with_integral(loc, scale):
    m = laplace.mean(loc, scale)
    q = noncentral_moment_with_integral(1, laplace, (loc, scale),
                                        support=(mp.ninf, loc, mp.inf),
                                        extradps=100)
    assert mp.almosteq(m, q)


@pytest.mark.parametrize('loc, scale',
                         [(0.25, 2.5), (-12, 3.5), (150, 123.5)])
@mp.workdps(50)
def test_var_with_integral(loc, scale):
    mu = laplace.mean(loc, scale)
    var = laplace.var(loc, scale)
    expected = mp.quad(lambda t: (t - mu)**2 * laplace.pdf(t, loc, scale),
                       [mp.ninf, loc, mp.inf])
    assert mp.almosteq(var, expected)


def test_mode():
    loc = 3
    m = laplace.mode(loc=loc, scale=2.5)
    assert m == loc


def test_support():
    sup = laplace.support(loc=-1, scale=9.5)
    assert sup == (mp.ninf, mp.inf)
