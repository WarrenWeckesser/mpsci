import pytest
from mpmath import mp
from mpsci.distributions import rice


@mp.workdps(55)
def test_pdf_logpdf():
    x = 5
    nu = 2
    sigma = 3
    p = rice.pdf(x, nu, sigma)
    # Reference value computed with Wolfram Alpha:
    #    PDF[RiceDistribution[2, 3], 5]
    val = '0.147895622189182736724316687543112924251334758218654595827535'
    assert mp.almosteq(p, mp.mpf(val))

    logp = rice.logpdf(x, nu, sigma)
    assert mp.almosteq(logp, mp.log(mp.mpf(val)))


@mp.workdps(55)
def test_cdf_sf():
    x = 5
    nu = 2
    sigma = 3
    cdf = rice.cdf(x, nu, sigma)
    # Reference value computed with Wolfram Alpha:
    #    CDF[RiceDistribution[2, 3], 5]
    val = '0.676335066267619908085529790477967022985718296948574400046663'
    assert mp.almosteq(cdf, mp.mpf(val))

    sf = rice.sf(x, nu, sigma)
    assert mp.almosteq(sf, 1 - mp.mpf(val))


@mp.workdps(55)
def test_mean():
    nu = 2
    sigma = 3
    m = rice.mean(nu, sigma)
    # Expected value computed with Wolfram Alpha:
    #   Mean[RiceDistribution[2, 3]]
    val = '4.16652436436299795861868470929154000443627136280337922061687'
    assert mp.almosteq(m, mp.mpf(val))


@pytest.mark.parametrize('nu, sigma',
                         [(0.5, 3.0),
                          (0.125, 25.0),
                          (10, 4),
                          (240, 725)])
def test_mean_with_integral(nu, sigma):
    m = rice.mean(nu, sigma)
    q = mp.quad(lambda t: t*rice.pdf(t, nu, sigma), [0, mp.inf])
    assert mp.almosteq(m, q)
