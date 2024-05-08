import pytest
from mpmath import mp
from mpsci.distributions import rel_breitwigner
from ._expect import (
    noncentral_moment_with_integral,
)


@pytest.mark.parametrize('rho', [0.125, 1, 8])
@mp.workdps(100)
def test_pdf_normalization(rho):
    rho = mp.mpf(rho)
    scale = mp.mpf(3.0)
    integral = mp.quad(lambda t: rel_breitwigner.pdf(t, rho, scale),
                       [0, rho*scale, mp.inf])
    assert mp.almosteq(integral, 1)


@pytest.mark.parametrize('x', ['0.1', 1, mp.inf])
@mp.workdps(100)
def test_cdf_with_quad(x):
    x = mp.mpf(x)
    rho = mp.mpf(0.5)
    scale = mp.mpf(2.5)
    cdf = rel_breitwigner.cdf(x, rho, scale)
    expected = mp.quad(lambda t: rel_breitwigner.pdf(t, rho, scale), [0, x])
    assert mp.almosteq(cdf, expected)


@pytest.mark.parametrize('rho', [0.125, 1, 8])
@pytest.mark.parametrize('x0', ['1e-12', 1, 25])
@mp.workdps(100)
def test_cdf_invcdf_roundtrip(rho, x0):
    x0 = mp.mpf(x0)
    scale = mp.mpf(1.25)
    p = rel_breitwigner.cdf(x0, rho, scale)
    x1 = rel_breitwigner.invcdf(p, rho, scale)
    assert mp.almosteq(x1, x0)


@pytest.mark.parametrize('x', ['0.1', 1, 100, 1e20])
@mp.workdps(100)
def test_sf_with_quad(x):
    x = mp.mpf(x)
    rho = mp.mpf(0.5)
    scale = mp.mpf(2.5)
    sf = rel_breitwigner.sf(x, rho, scale)
    expected = mp.quad(lambda t: rel_breitwigner.pdf(t, rho, scale),
                       [x, mp.inf])
    assert mp.almosteq(sf, expected)


@pytest.mark.parametrize('rho', [0.125, 1, 8])
@pytest.mark.parametrize('x0', ['1e-12', 1, 25])
@mp.workdps(100)
def test_sf_invsf_roundtrip(rho, x0):
    x0 = mp.mpf(x0)
    scale = mp.mpf(1.25)
    p = rel_breitwigner.sf(x0, rho, scale)
    x1 = rel_breitwigner.invsf(p, rho, scale)
    assert mp.almosteq(x1, x0)


@pytest.mark.parametrize('rho, scale', [(1.125, 2),
                                        (5.0, 0.125)])
@mp.workdps(50)
def test_mean_with_quad(rho, scale):
    mean = rel_breitwigner.mean(rho, scale)
    intgrl = noncentral_moment_with_integral(1, rel_breitwigner, (rho, scale),
                                             extradps=2*mp.dps)
    assert mp.almosteq(mean, intgrl), f"{mean} not almost equal to {intgrl}"


@pytest.mark.parametrize('rho, scale', [(1.125, 2),
                                        (5.0, 0.125)])
@mp.workdps(50)
def test_noncentral_moment2_with_quad(rho, scale):
    mu2 = rel_breitwigner.noncentral_moment(2, rho, scale)
    intgrl = noncentral_moment_with_integral(2, rel_breitwigner, (rho, scale),
                                             extradps=2*mp.dps)
    assert mp.almosteq(mu2, intgrl)
