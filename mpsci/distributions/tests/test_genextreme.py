import pytest
from mpmath import mp
from mpsci.distributions import genextreme


@mp.workdps(50)
def test_basic_pdf():
    # The expected values were computed "by hand".
    expected = mp.exp(mp.mpf('-0.5'))/16
    assert mp.almosteq(genextreme.pdf(6, 2, 3, 2), expected)
    expected = mp.exp(mp.mpf('-2'))/4
    assert mp.almosteq(genextreme.pdf(0, -2, 3, 2), expected)


@mp.workdps(50)
def test_basic_logpdf():
    # The expected values were computed "by hand".
    expected = mp.mpf('-0.5') - mp.log(16)
    assert mp.almosteq(genextreme.logpdf(6, 2, 3, 2), expected)
    expected = mp.mpf('-2') - mp.log(4)
    assert mp.almosteq(genextreme.logpdf(0, -2, 3, 2), expected)


@mp.workdps(50)
def test_basic_cdf():
    # The expected value was computed "by hand".
    expected = mp.exp(mp.mpf('-0.5'))
    assert mp.almosteq(genextreme.cdf(6, 2, 3, 2), expected)


@mp.workdps(50)
def test_basic_sf():
    # The expected value was computed "by hand".
    expected = 1 - mp.exp(mp.mpf('-0.5'))
    assert mp.almosteq(genextreme.sf(6, 2, 3, 2), expected)


@pytest.mark.parametrize('x, xi, mu, sigma',
                         [(3.0, 0.5, -3, 0.25),
                          (-0.5, 1.5, 0.5, 4.0),
                          (-2.0, 0.0, 5.0, 3.0),
                          (0.0, -0.5, 2, 1.5),
                          (1.5, -2.0, 2, 1.5)])
@mp.workdps(50)
def test_cdf_invcdf_roundtrip(x, xi, mu, sigma):
    p = genextreme.cdf(x, xi, mu, sigma)
    x1 = genextreme.invcdf(p, xi, mu, sigma)
    assert mp.almosteq(x1, x)


@pytest.mark.parametrize('x, xi, mu, sigma',
                         [(3.0, 0.5, -3, 0.25),
                          (-0.5, 1.5, 0.5, 4.0),
                          (-2.0, 0.0, 5.0, 3.0),
                          (0.0, -0.5, 2, 1.5),
                          (1.5, -2.0, 2, 1.5)])
@mp.workdps(50)
def test_sf_invsf_roundtrip(x, xi, mu, sigma):
    p = genextreme.sf(x, xi, mu, sigma)
    x1 = genextreme.invsf(p, xi, mu, sigma)
    assert mp.almosteq(x1, x)


@mp.workdps(50)
def test_mean():
    xi = mp.mpf('0.5')
    mu = 3
    sigma = 2
    g1 = mp.gamma(1 - xi)
    assert mp.almosteq(genextreme.mean(xi, mu, sigma),
                       mu + sigma * (g1 - 1)/xi)


@mp.workdps(50)
def test_mean_xi_zero():
    xi = 0
    mu = 3
    sigma = 2
    assert mp.almosteq(genextreme.mean(xi, mu, sigma),
                       mu + sigma * mp.euler)


@mp.workdps(50)
def test_inf_mean():
    assert genextreme.mean(2, 3, 2) == mp.inf


@mp.workdps(50)
def test_var():
    xi = mp.mpf('0.25')
    mu = 3
    sigma = 2
    g1 = mp.gamma(1 - xi)
    g2 = mp.gamma(1 - 2*xi)
    assert mp.almosteq(genextreme.var(xi, mu, sigma),
                       sigma**2 * (g2 - g1**2) / xi**2)


@mp.workdps(50)
def test_var_xi_zero():
    xi = 0
    mu = 3
    sigma = 2
    assert mp.almosteq(genextreme.var(xi, mu, sigma),
                       sigma**2 * mp.pi**2 / 6)


@mp.workdps(50)
def test_inf_var():
    assert genextreme.var(2, 3, 2) == mp.inf
