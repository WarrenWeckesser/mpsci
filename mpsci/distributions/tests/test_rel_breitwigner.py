import pytest
from mpmath import mp
from mpsci.distributions import rel_breitwigner


mp.dps = 100


@pytest.mark.parametrize('rho', [0.125, 1, 8])
def test_pdf_normalization(rho):
    rho = mp.mpf(rho)
    scale = mp.mpf(3.0)
    integral = mp.quad(lambda t: rel_breitwigner.pdf(t, rho, scale),
                       [0, rho*scale, mp.inf])
    assert mp.almosteq(integral, 1)


@pytest.mark.parametrize('x', [0.1, 1, mp.inf])
def test_cdf_with_quad(x):
    rho = mp.mpf(0.5)
    scale = mp.mpf(2.5)
    cdf = rel_breitwigner.cdf(x, rho, scale)
    expected = mp.quad(lambda t: rel_breitwigner.pdf(t, rho, scale), [0, x])
    assert mp.almosteq(cdf, expected)


@pytest.mark.parametrize('rho', [0.125, 1, 8])
@pytest.mark.parametrize('x0', [mp.mpf('1e-12'), 1, 25])
def test_cdf_invcdf_roundtrip(rho, x0):
    scale = mp.mpf(1.25)
    p = rel_breitwigner.cdf(x0, rho, scale)
    x1 = rel_breitwigner.invcdf(p, rho, scale)
    assert mp.almosteq(x1, x0)


@pytest.mark.parametrize('rho, scale', [(mp.mpf(1.125), mp.mpf(2)),
                                        (mp.mpf(5.0), mp.mpf(0.125))])
def test_mean_with_quad(rho, scale):
    mean = rel_breitwigner.mean(rho, scale)
    integral = mp.quad(lambda t: t*rel_breitwigner.pdf(t, rho, scale),
                       [0, rho*scale, mp.inf])
    assert mp.almosteq(mean, integral)


@pytest.mark.parametrize('rho, scale', [(mp.mpf(1.125), mp.mpf(2)),
                                        (mp.mpf(5.0), mp.mpf(0.125))])
def test_noncentral_moment2_with_quad(rho, scale):
    mu2 = rel_breitwigner.noncentral_moment(2, rho, scale)
    integral = mp.quad(lambda t: t**2*rel_breitwigner.pdf(t, rho, scale),
                       [0, rho*scale, mp.inf])
    assert mp.almosteq(mu2, integral)
