import pytest
from mpmath import mp
from mpsci.distributions import half_logistic, logistic


def test_support():
    loc = 3.0
    scale = 2.5
    sup = half_logistic.support(loc, scale)
    assert sup == (loc, mp.inf)


def test_pdf_bad_scale():
    with pytest.raises(ValueError, match='must be greater than 0'):
        half_logistic.pdf(3, 2, -0.5)


def test_pdf_out_of_support():
    assert half_logistic.pdf(-1, 0, 1) == 0
    assert half_logistic.logpdf(-1, 0, 1) == mp.ninf


def test_cdf_sf_out_of_support():
    assert half_logistic.cdf(-1, 0, 1) == 0
    assert half_logistic.sf(-1, 0, 1) == 1


@mp.workdps(40)
def test_pdf_logpdf_consistency():
    x = 10.0
    loc = 1
    scale = 2.5
    logp = half_logistic.logpdf(x, loc, scale)
    with mp.extradps(5):
        log_pdf = mp.log(half_logistic.pdf(x, loc, scale))
    assert mp.almosteq(logp, log_pdf)


@mp.workdps(40)
@pytest.mark.parametrize(
    'x, loc, scale',
    [(0.0, 0, 1), (1.0, 0, 1), (2.5, -1, 3), (100, 2, 0.5)]
)
def test_pdf_against_logistic(x, loc, scale):
    p = half_logistic.pdf(x, loc, scale)
    p_logistic = logistic.pdf(x, loc, scale)
    assert mp.almosteq(p, 2*p_logistic)


@mp.workdps(40)
@pytest.mark.parametrize(
    'x, loc, scale',
    [(0.0, 0, 1), (1.0, 0, 1), (2.5, -1, 3), (100, 2, 0.5)]
)
def test_cdf_against_logistic(x, loc, scale):
    cdf = half_logistic.cdf(x, loc, scale)
    cdf_logistic = logistic.cdf(x, loc, scale)
    assert mp.almosteq(cdf, 2*cdf_logistic - 1)


@mp.workdps(40)
@pytest.mark.parametrize(
    'x, loc, scale',
    [(0.0, 0, 1), (1.0, 0, 1), (2.5, -1, 3), (100, 2, 0.5)]
)
def test_sf_against_logistic(x, loc, scale):
    sf = half_logistic.sf(x, loc, scale)
    sf_logistic = logistic.sf(x, loc, scale)
    assert mp.almosteq(sf, 2*sf_logistic)


@mp.workdps(40)
def test_invcdf_against_logistic():
    p = mp.mpf(0.25)
    loc = 1
    scale = 4
    invcdf = half_logistic.invcdf(p, loc, scale)
    assert mp.almosteq(invcdf, logistic.invcdf((p+1)/2, loc, scale))


@mp.workdps(40)
def test_invsf_against_logistic():
    p = mp.mpf(0.25)
    loc = 1
    scale = 4
    invsf = half_logistic.invsf(p, loc, scale)
    assert mp.almosteq(invsf, logistic.invsf(p/2, loc, scale))


def test_invcdf_edges():
    loc = 1.0
    scale = 4.0
    x = half_logistic.invcdf(0, loc, scale)
    assert x == loc
    x = half_logistic.invcdf(1, loc, scale)
    assert x == mp.inf


def test_invsf_edges():
    loc = 1.0
    scale = 4.0
    x = half_logistic.invsf(0, loc, scale)
    assert x == mp.inf
    x = half_logistic.invsf(1, loc, scale)
    assert x == loc
