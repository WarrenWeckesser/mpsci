import pytest
from mpmath import mp
from mpsci.distributions import folded_normal
from ._expect import expect


def test_bad_sigma():
    with pytest.raises(ValueError, match='must be positive'):
        folded_normal.pdf(3, 4, -5)


def test_support():
    mu = 1
    sigma = 3
    sup = folded_normal.support(mu, sigma)
    assert sup == (0, mp.inf)


def test_pdf_logpdf_out_of_support():
    mu = 2
    sigma = 8
    x = -1
    assert folded_normal.pdf(x, mu, sigma) == 0
    assert folded_normal.logpdf(x, mu, sigma) == mp.ninf


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


def test_cdf_sf_out_of_support():
    mu = 2
    sigma = 8
    x = -1
    assert folded_normal.cdf(x, mu, sigma) == 0
    assert folded_normal.sf(x, mu, sigma) == 1


@mp.workdps(80)
def test_mean_with_integral():
    mu = 5
    sigma = 0.5
    m = folded_normal.mean(mu, sigma)
    expected = expect(folded_normal, (mu, sigma), lambda t: t,
                      support=[0, mp.inf])
    assert mp.almosteq(m, expected)


@mp.workdps(80)
def test_var_with_integral():
    mu = 5
    sigma = 0.5
    mean = folded_normal.mean(mu, sigma)
    v = folded_normal.var(mu, sigma)
    expected = expect(folded_normal, (mu, sigma), lambda t: (t - mean)**2,
                      support=[0, mp.inf])
    assert mp.almosteq(v, expected)
