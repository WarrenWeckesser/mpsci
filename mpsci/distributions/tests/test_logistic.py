import pytest
from mpmath import mp
from mpsci.distributions import logistic


@pytest.mark.parametrize('x', [-20, -1, 0, 3, 18])
def test_pdf(x):
    with mp.workdps(40):
        loc = mp.mpf(2)
        scale = mp.mpf(5)
        p = logistic.pdf(x, loc=loc, scale=scale)
        t = (x - loc)/scale/2
        expected_p = 1/(mp.exp(t) + mp.exp(-t))**2 / scale
        assert mp.almosteq(p, expected_p)


@pytest.mark.parametrize('x', [-20, -1, 0, 3, 18])
def test_cdf(x):
    with mp.workdps(40):
        loc = mp.mpf(2)
        scale = mp.mpf(5)
        p = logistic.cdf(x, loc=loc, scale=scale)
        t = (x - loc)/scale
        expected_p = 1/(1 + mp.exp(-t))
        assert mp.almosteq(p, expected_p)


@pytest.mark.parametrize('x', [-20, -1, 0, 3, 18])
def test_sf(x):
    with mp.workdps(40):
        loc = mp.mpf(2)
        scale = mp.mpf(5)
        p = logistic.sf(x, loc=loc, scale=scale)
        ex = mp.exp(-(x - loc)/scale)
        expected_p = ex/(1 + ex)
        assert mp.almosteq(p, expected_p)


@pytest.mark.parametrize('p', [0.125, 0.25, 0.5, 0.75, 0.875])
def test_invcdf(p):
    with mp.workdps(40):
        p = mp.mpf(p)
        loc = mp.mpf(2)
        scale = mp.mpf(5)
        x = logistic.invcdf(p, loc=loc, scale=scale)
        expected_x = loc + scale*mp.log(p/(1 - p))
        assert mp.almosteq(x, expected_x)


@pytest.mark.parametrize('p', [0.125, 0.25, 0.5, 0.75, 0.875])
def test_invsf(p):
    with mp.workdps(40):
        p = mp.mpf(p)
        loc = mp.mpf(2)
        scale = mp.mpf(5)
        x = logistic.invsf(p, loc=loc, scale=scale)
        expected_x = loc + scale*mp.log((1 - p)/p)
        assert mp.almosteq(x, expected_x)


@pytest.mark.parametrize('scale', [1, 0.5])
def test_entropy(scale):
    ent = logistic.entropy(scale=scale)
    assert mp.almosteq(ent, mp.log(scale) + 2)
