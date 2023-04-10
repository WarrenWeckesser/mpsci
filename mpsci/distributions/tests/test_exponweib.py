import pytest
from mpmath import mp
from mpsci.distributions import exponweib


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
