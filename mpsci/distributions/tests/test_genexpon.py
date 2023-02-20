import pytest
from mpmath import mp
from mpsci.distributions import genexpon


@pytest.mark.parametrize('a, b, c', [(2, 3, 4), (10, 0.03125, 2)])
def test_genexpon_pdf_is_normalized(a, b, c):
    with mp.workdps(80):
        i = mp.quad(lambda t: genexpon.pdf(t, a, b, c), [0, mp.inf])
        # Testing for equality is probably too optimistic...
        assert i == 1


@pytest.mark.parametrize('fun, invfun', [(genexpon.cdf, genexpon.invcdf),
                                         (genexpon.sf, genexpon.invsf)])
@pytest.mark.parametrize('x, a, b, c', [(0.0625, 2, 3, 4), (0.125, 2, 3, 4),
                                        (0.75, 2, 3, 4), (4, 2, 3, 4),
                                        (1e-3, 10, 0.0625, 2),
                                        (1e-1, 10, 0.0625, 2),
                                        (2.0, 10, 0.0625, 2)])
def test_fun_invfun_roundtrip(fun, invfun, x, a, b, c):
    with mp.workdps(80):
        p = fun(x, a, b, c)
        x2 = invfun(p, a, b, c)
        assert mp.almosteq(x2, x, rel_eps=2**(-mp.prec + 8), abs_eps=0)
