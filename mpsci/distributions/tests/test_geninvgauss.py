import pytest
from mpmath import mp
from mpsci.distributions import geninvgauss as gig
from ._expect import (
    check_entropy_with_integral,
    noncentral_moment_with_integral,
)


@pytest.mark.parametrize('p, b, loc, scale',
                         [(-1, 0.5, 0, 0.25),
                          (0.5, 1, 0, 1),
                          (2, 3, 0, 5)])
@mp.workdps(20)
def test_entropy_against_integral(p, b, loc, scale):
    check_entropy_with_integral(gig, (p, b, loc, scale))


@mp.workdps(50)
def test_var_with_integral():
    p = 1.5
    b = 2.5
    loc = 3.5
    scale = 6
    var = gig.var(p, b, loc, scale)

    mean = gig.mean(p, b, loc, scale)
    mom2 = noncentral_moment_with_integral(2, gig, (p, b, loc, scale))
    assert mp.almosteq(var, mom2 - mean**2)
