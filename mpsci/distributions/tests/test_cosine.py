
import pytest
from mpmath import mp
from mpsci.distributions import cosine


@mp.workdps(50)
def test_support():
    a, b = cosine.support()
    assert mp.almosteq(a, -mp.pi)
    assert mp.almosteq(b, mp.pi)


@mp.workdps(25)
def test_pdf_logpdf_out_of_support():
    assert cosine.pdf(-3.5) == 0
    assert cosine.pdf(3.25) == 0
    assert cosine.logpdf(-3.5) == mp.ninf
    assert cosine.logpdf(3.25) == mp.ninf


@mp.workdps(25)
def test_cdf_sf_out_of_support():
    assert cosine.cdf(-3.5) == 0
    assert cosine.cdf(3.25) == 1
    assert cosine.sf(-3.5) == 1
    assert cosine.sf(3.25) == 0


@mp.workdps(60)
def test_pdf_normalization():
    integral = mp.quad(cosine.pdf, [-mp.pi, mp.pi])
    assert mp.almosteq(integral, 1)


@pytest.mark.parametrize(
    'x', ['-3.14159265', -3, -0.25, 0, 1.25, 2.75, '3.14159265']
)
@mp.workdps(50)
def test_pdf_logpdf_consistency(x):
    x = mp.mpf(x)
    log_pdf = mp.log(cosine.pdf(x))
    logpdf = cosine.logpdf(x)
    assert mp.almosteq(logpdf, log_pdf)


@pytest.mark.parametrize('funcpair', [(cosine.cdf, cosine.invcdf),
                                      (cosine.sf, cosine.invsf)])
@pytest.mark.parametrize(
    'x0',
    [-mp.pi, mp.mpf('-3.14159'), -3.0, 0, 0.25, 3.0, mp.mpf('3.14159'), mp.pi])
@mp.workdps(50)
def test_dist_roundtrip(funcpair, x0):
    func, invfunc = funcpair
    with mp.extradps(25):
        p = func(x0)
        x1 = invfunc(p)
    assert mp.almosteq(x1, x0, rel_eps=2**(-mp.prec + 10), abs_eps=0)


@mp.workdps(25)
def test_mean():
    assert cosine.mean() == 0
