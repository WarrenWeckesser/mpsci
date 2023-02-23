
import pytest
from mpmath import mp
from mpsci.distributions import genpareto


@pytest.mark.parametrize('func, invfunc',
                         [(genpareto.cdf, genpareto.invcdf),
                          (genpareto.sf, genpareto.invsf)])
@pytest.mark.parametrize('x, xi, mu, sigma',
                         [(8, 1.5, 2, 3.0),
                          (3, 0.5, 2.5, 10.0),
                          (3, 0, 2, 3.0),
                          (3, -2, 1.5, 5.0)])
def test_cdf_invcdf_sf_invsf_roundtrip(func, invfunc, x, xi, mu, sigma):
    p = func(x, xi, mu, sigma)
    x1 = invfunc(p, xi, mu, sigma)
    assert mp.almosteq(x1, x)
