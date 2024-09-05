import pytest
from mpmath import mp
from mpsci.distributions import exponweib


def test_support():
    a = 1.25
    c = 2.75
    scale = 3
    sup = exponweib.support(a, c, scale=scale)
    assert sup == (mp.zero, mp.inf)


def test_bad_params():
    with pytest.raises(ValueError, match='`a` must be greater than 0'):
        exponweib.pdf(1, 0, 2, 3)
    with pytest.raises(ValueError, match='`c` must be greater than 0'):
        exponweib.pdf(1, 1, 0, 3)
    with pytest.raises(ValueError, match='`scale` must be greater than 0'):
        exponweib.pdf(1, 1, 2, -3)


def test_pdf_logpdf_out_of_support():
    a = 3.25
    b = 0.25
    scale = 0.125
    x = -1.5
    p = exponweib.pdf(x, a, b, scale)
    assert p == 0
    logp = exponweib.logpdf(x, a, b, scale)
    assert logp == mp.ninf


@mp.workdps(60)
def test_pdf_normalization():
    integral = mp.quad(lambda t: exponweib.pdf(t, 1.25, 2, scale=0.5),
                       [0, 0.5, mp.inf],
                       method='tanh-sinh')
    assert mp.almosteq(integral, 1)


@pytest.mark.parametrize('x', ['1e-250', 8, '1e6'])
@mp.workdps(60)
def test_pdf_logpdf_consistency(x):
    # Consistency check: logpdf(x) ~= log(pdf(x))
    a = 1.25
    c = 2.75
    scale = 3
    x = mp.mpf(x)
    logp1 = exponweib.logpdf(x, a, c, scale=scale)
    with mp.extradps(5):
        logp2 = mp.log(exponweib.pdf(x, a, c, scale=scale))
    assert mp.almosteq(logp1, logp2)


def test_cf_sf_out_of_support():
    a = 3.25
    b = 0.25
    scale = 0.125
    x = -1.5
    p = exponweib.cdf(x, a, b, scale)
    assert p == 0
    q = exponweib.sf(x, a, b, scale)
    assert q == 1


@pytest.mark.parametrize(
    'x, a, c, scale',
    [(3, 10, 0.5, 2.5),
     (100, 1, 2, 3)]
)
@mp.workdps(50)
def test_cdf_sf_consistency(x, a, c, scale):
    # Test that CDF + SF is 1.
    cdf = exponweib.cdf(x, a, c, scale=scale)
    sf = exponweib.sf(x, a, c, scale=scale)
    assert mp.almosteq(cdf + sf, mp.one)


@pytest.mark.parametrize(
    'x, a, c, scale',
    [(1, 2, 3, 5),
     (0.5, 10, 0.25, 0.5),
     (3, 0.5, 5, 2)]
)
@mp.workdps(150)
def test_cdf_invcdf_roundtrip(x, a, c, scale):
    # This checks the roundtrip x = invcdf(cdf(x)).
    cdf = exponweib.cdf(x, a, c, scale)
    x1 = exponweib.invcdf(cdf, a, c, scale)
    assert mp.almosteq(x1, x)


@pytest.mark.parametrize(
    'x, a, c, scale',
    [(1, 2, 3, 5),
     (0.5, 10, 0.25, 0.5),
     (3, 0.5, 5, 2)]
)
@mp.workdps(150)
def test_sf_invsf_roundtrip(x, a, c, scale):
    # This checks the roundtrip x = invsf(sf(x)).
    sf = exponweib.sf(x, a, c, scale)
    x1 = exponweib.invsf(sf, a, c, scale)
    assert mp.almosteq(x1, x)
