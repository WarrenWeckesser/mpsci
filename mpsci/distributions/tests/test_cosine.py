
import pytest
from mpmath import mp
from mpsci.distributions import cosine


mp.dps = 80


@pytest.mark.parametrize('funcpair', [(cosine.cdf, cosine.invcdf),
                                      (cosine.sf, cosine.invsf)])
@pytest.mark.parametrize(
    'x0',
    [-mp.pi, mp.mpf('-3.14159'), -3.0, 0, 0.25, 3.0, mp.mpf('3.14159'), mp.pi])
def test_dist_roundtrip(funcpair, x0):
    func, invfunc = funcpair
    p = func(x0)
    x1 = invfunc(p)
    assert mp.almosteq(x1, x0, rel_eps=2**(-mp.prec+24), abs_eps=0)
