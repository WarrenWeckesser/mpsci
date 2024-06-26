import pytest
from mpmath import mp
from mpsci.distributions import slash


@mp.workdps(60)
def test_pdf_normalization():
    integral = mp.quad(slash.pdf, [-mp.inf, -3, 0, 3, mp.inf],
                       method='gauss-legendre')
    assert mp.almosteq(integral, 1)


@pytest.mark.parametrize('x', [-3, 0, 13])
@mp.workdps(60)
def test_pdf_logpdf(x):
    # Consistency check: logpdf(x) ~= log(pdf(x))
    logp1 = slash.logpdf(x)
    logp2 = mp.log(slash.pdf(x))
    assert mp.almosteq(logp1, logp2)


@mp.workdps(50)
def test_pdf_small_x():
    x = mp.mpf('1/100000000')
    p = slash.pdf(x)
    # Wolfram Alpha:
    #    (PDF[NormalDistribution[0, 1], 0]
    #     - PDF[NormalDistribution[0, 1], 1/100000000])
    #    /(1/100000000)^2
    valstr = '0.19947114020071633398319452494928254310157865003450084'
    ref = mp.mpf(valstr)
    assert mp.almosteq(p, ref)


@mp.workdps(50)
def test_cdf_small_x():
    x = mp.mpf('1/100000000')
    c = slash.cdf(x)
    # Wolfram Alpha:
    #    CDF[NormalDistribution[0, 1], 1/100000000] -
    #    (PDF[NormalDistribution[0, 1], 0]
    #     - PDF[NormalDistribution[0, 1], 1/100000000])
    #    /(1/100000000)
    valstr = '0.50000000199471140200716337307713528294554792777415749'
    ref = mp.mpf(valstr)
    assert mp.almosteq(c, ref)


@pytest.mark.parametrize('x', [-3, 0, 1, 25])
@mp.workdps(50)
def test_cdf_with_quad(x):
    cdf = slash.cdf(x)
    if x > 0:
        points = [-mp.inf, 0, x]
    else:
        points = [-mp.inf, x]
    integral = mp.quad(slash.pdf, points, method='gauss-legendre')
    assert mp.almosteq(cdf, integral)


@pytest.mark.parametrize('x', [-3, 0, 1, 25])
@mp.workdps(50)
def test_sf_with_quad(x):
    sf = slash.sf(x)
    if x < 0:
        points = [x, 0, mp.inf]
    else:
        points = [x, mp.inf]
    integral = mp.quad(slash.pdf, points, method='gauss-legendre')
    assert mp.almosteq(sf, integral)


@pytest.mark.parametrize('x', [-mp.inf, -3, 0, 1, 25, mp.inf])
@mp.workdps(50)
def test_cdf_invcdf_roundtrip(x):
    p = slash.cdf(x)
    x2 = slash.invcdf(p)
    if mp.isinf(x):
        assert x2 == x
    else:
        assert mp.almosteq(x2, x)


@pytest.mark.parametrize('x', [-mp.inf, -3, 0, 1, 25, mp.inf])
@mp.workdps(50)
def test_sf_invsf_roundtrip(x):
    p = slash.sf(x)
    x2 = slash.invsf(p)
    if mp.isinf(x):
        assert x2 == x
    else:
        assert mp.almosteq(x2, x)
