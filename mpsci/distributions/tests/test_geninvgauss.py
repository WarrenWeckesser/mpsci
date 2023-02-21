import pytest
from mpmath import mp
from mpsci.distributions import geninvgauss as gig


@pytest.mark.parametrize('p, b, scale', [(-1, 0.5, 0.25), (0.5, 1, 1), (2, 3, 5)])
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
