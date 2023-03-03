
import pytest
from mpmath import mp
from mpsci.distributions import power_normal


@mp.workdps(40)
@pytest.mark.parametrize('x, c, loc, scale',
                         [(-1.5, 2.75, 1, 3.5),
                          (10, 1, 0, 3),
                          (0, 9, 0, 0.25)])
def test_pdf_logpdf(x, c, loc, scale):
    # This is a test for consistency of pdf() and logpdf().
    # It does not check that the values are the actual correct
    # values for the power normal distribution.
    p = power_normal.pdf(x, c, loc, scale)
    logp = power_normal.logpdf(x, c, loc, scale)
    assert mp.almosteq(logp, mp.log(p))


@mp.workdps(40)
@pytest.mark.parametrize('x, c, loc, scale',
                         [(-1.5, 2.75, 1, 3.5),
                          (10, 1, 0, 3)])
def test_cdf_invcdf_sf_invsf_roundtrip(x, c, loc, scale):
    cdf = power_normal.cdf(x, c, loc=loc, scale=scale)
    x1 = power_normal.invcdf(cdf, c, loc=loc, scale=scale)
    assert mp.almosteq(x, x1)

    sf = power_normal.sf(x, c, loc=loc, scale=scale)
    x1 = power_normal.invsf(sf, c, loc=loc, scale=scale)
    assert mp.almosteq(x, x1)


@mp.workdps(40)
@pytest.mark.parametrize('x, c, loc, scale',
                         [(-1.5, 2.75, 1, 3.5),
                          (10, 1, 0, 3),
                          (2, 0.25, 0, 5)])
def test_cdf_sf_consistency(x, c, loc, scale):
    cdf = power_normal.cdf(x, c, loc=loc, scale=scale)
    sf = power_normal.sf(x, c, loc=loc, scale=scale)
    # This should be true for x values are that not too far into
    # the tails of the distribution.
    assert mp.almosteq(sf, 1 - cdf)
