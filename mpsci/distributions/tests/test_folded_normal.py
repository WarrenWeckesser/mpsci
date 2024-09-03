
from mpmath import mp
from mpsci.distributions import folded_normal


@mp.workdps(80)
def test_pdf():
    mu = 1
    sigma = 3
    x = 5
    p = folded_normal.pdf(x, mu, sigma)
    expected = mp.npdf(x, mu, sigma) + mp.npdf(-x, mu, sigma)
    assert mp.almosteq(p, expected)


@mp.workdps(80)
def test_logpdf():
    mu = 1
    sigma = 3
    x = 5
    p = folded_normal.pdf(x, mu, sigma)
    expected = mp.log(p)
    logp = folded_normal.logpdf(x, mu, sigma)
    assert mp.almosteq(logp, expected)


@mp.workdps(80)
def test_cdf():
    mu = 1
    sigma = 3
    x = 5
    cdf = folded_normal.cdf(x, mu, sigma)
    expected = mp.ncdf(x, mu, sigma) - mp.ncdf(-x, mu, sigma)
    assert mp.almosteq(cdf, expected)


@mp.workdps(80)
def test_sf():
    mu = 1
    sigma = 3
    x = 5
    sf = folded_normal.sf(x, mu, sigma)
    # expected = mp.one - (mp.ncdf(x, mu, sigma) - mp.ncdf(-x, mu, sigma))
    expected = mp.ncdf(-x, -mu, sigma) + mp.ncdf(-x, mu, sigma)
    assert mp.almosteq(sf, expected)
