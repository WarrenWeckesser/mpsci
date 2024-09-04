import pytest
from mpmath import mp
from mpsci.distributions import maxwell


@mp.workdps(50)
def test_pdf():
    x = 1.5
    scale = 3
    y = maxwell.pdf(x, loc=0, scale=scale)
    # The reference value was computed with Wolfram Alpha:
    #   PDF[MaxwellDistribution[3], 3/2]
    val = '0.058677554460716579629113406932752942185052530062618658'
    assert mp.almosteq(y, mp.mpf(val))


@mp.workdps(50)
def test_logpdf():
    x = 1.5
    scale = 3
    y = maxwell.logpdf(x, loc=0, scale=scale)
    # The reference value was computed with Wolfram Alpha:
    #   Log[PDF[MaxwellDistribution[3], 3/2]]
    val = '-2.8356980024327277425928070947863199125843881658207881'
    assert mp.almosteq(y, mp.mpf(val))


@mp.workdps(50)
def test_cdf():
    x = 1.5
    scale = 3
    y = maxwell.cdf(x, loc=0, scale=scale)
    # The reference value was computed with Wolfram Alpha:
    #   CDF[MaxwellDistribution[3], 3/2]
    val = '0.030859595783726729500728779620157826656889170733443923986'
    assert mp.almosteq(y, mp.mpf(val))


@mp.workdps(50)
def test_sf():
    x = 1.5
    scale = 3
    y = maxwell.sf(x, loc=0, scale=scale)
    # The reference value was computed with Wolfram Alpha:
    #   1 - CDF[MaxwellDistribution[3], 3/2]
    val = '0.96914040421627327049927122037984217334311082926655607601'
    assert mp.almosteq(y, mp.mpf(val))


@pytest.mark.parametrize('x', [1/1024, 1/128, 0.125, 0.5, 0.9921875])
@mp.workdps(150)
def test_cdf_invcdf_roundtrip(x):
    loc = 0
    scale = 2.5
    p = maxwell.cdf(x, loc, scale)
    x1 = maxwell.invcdf(p, loc, scale)
    assert mp.almosteq(x1, x)


@pytest.mark.parametrize('x', [1/128, 0.125, 0.5, 0.9921875])
@mp.workdps(150)
def test_sf_invsf_roundtrip(x):
    loc = 0
    scale = 2.5
    p = maxwell.sf(x, loc, scale)
    x1 = maxwell.invsf(p, loc, scale)
    assert mp.almosteq(x1, x)
