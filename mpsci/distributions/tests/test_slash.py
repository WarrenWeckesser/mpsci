import pytest
from mpmath import mp
from mpsci.distributions import slash


mp.dps = 60


def test_pdf_normalization():
    integral = mp.quad(slash.pdf, [-mp.inf, -3, 0, 3, mp.inf],
                       method='gauss-legendre')
    assert mp.almosteq(integral, 1)


@pytest.mark.parametrize('x', [-3, 0, 13])
def test_pdf_logpdf(x):
    # Consistency check: logpdf(x) ~= log(pdf(x))
    logp1 = slash.logpdf(x)
    logp2 = mp.log(slash.pdf(x))
    assert mp.almosteq(logp1, logp2)


@pytest.mark.parametrize('x', [-3, 0, 1, 25])
def test_cdf_with_quad(x):
    cdf = slash.cdf(x)
    if x > 0:
        points = [-mp.inf, 0, x]
    else:
        points = [-mp.inf, x]
    integral = mp.quad(slash.pdf, points, method='gauss-legendre')
    assert mp.almosteq(cdf, integral)


@pytest.mark.parametrize('x', [-3, 0, 1, 25])
def test_sf_with_quad(x):
    sf = slash.sf(x)
    if x < 0:
        points = [x, 0, mp.inf]
    else:
        points = [x, mp.inf]
    integral = mp.quad(slash.pdf, points, method='gauss-legendre')
    assert mp.almosteq(sf, integral)


@pytest.mark.parametrize('x', [-mp.inf, -3, 0, 1, 25, mp.inf])
def test_cdf_invcdf_roundtrip(x):
    p = slash.cdf(x)
    x2 = slash.invcdf(p)
    if mp.isinf(x):
        assert x2 == x
    else:
        assert mp.almosteq(x2, x)


@pytest.mark.parametrize('x', [-mp.inf, -3, 0, 1, 25, mp.inf])
def test_sf_invsf_roundtrip(x):
    p = slash.sf(x)
    x2 = slash.invsf(p)
    if mp.isinf(x):
        assert x2 == x
    else:
        assert mp.almosteq(x2, x)
