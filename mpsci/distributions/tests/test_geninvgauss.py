import pytest
from mpmath import mp
from mpsci.distributions import geninvgauss as gig


@pytest.mark.parametrize('p, b, scale',
                         [(-1, 0.5, 0.25),
                          (0.5, 1, 1),
                          (2, 3, 5)])
def test_entropy_against_integral(p, b, scale):
    with mp.workdps(25):
        entr = gig.entropy(p, b, scale=scale)
        mode = gig.mode(p, b, scale=scale)
        if mode == 0:
            pts = [0, mp.inf]
        else:
            pts = [0, mode, mp.inf]
        q = -mp.quad(lambda t: (gig.logpdf(t, p, b, scale=scale) *
                                gig.pdf(t, p, b, scale=scale)),
                     pts)
        assert mp.almosteq(entr, q)


@mp.workdps(50)
def test_var_with_integral():
    p = 1.5
    b = 2.5
    loc = 3.5
    scale = 6
    var = gig.var(p, b, loc, scale)

    mean = gig.mean(p, b, loc, scale)
    mom2 = mp.quad(lambda t: t**2*gig.pdf(t, p, b, loc, scale), [loc, mp.inf])
    assert mp.almosteq(var, mom2 - mean**2)
