import pytest
from mpmath import mp
from mpsci.distributions import logistic


@pytest.mark.parametrize('x', [-20, -1, 0, 3, 18])
@mp.workdps(50)
def test_pdf(x):
    loc = mp.mpf(2)
    scale = mp.mpf(5)
    p = logistic.pdf(x, loc=loc, scale=scale)
    t = (x - loc)/scale/2
    expected_p = 1/(mp.exp(t) + mp.exp(-t))**2 / scale
    assert mp.almosteq(p, expected_p)


@pytest.mark.parametrize('x', [-20, -1, 0, 3, 18])
@mp.workdps(50)
def test_cdf(x):
    loc = mp.mpf(2)
    scale = mp.mpf(5)
    p = logistic.cdf(x, loc=loc, scale=scale)
    t = (x - loc)/scale
    expected_p = 1/(1 + mp.exp(-t))
    assert mp.almosteq(p, expected_p)


@pytest.mark.parametrize('x', [-20, -1, 0, 3, 18])
@mp.workdps(50)
def test_sf(x):
    loc = mp.mpf(2)
    scale = mp.mpf(5)
    p = logistic.sf(x, loc=loc, scale=scale)
    ex = mp.exp(-(x - loc)/scale)
    expected_p = ex/(1 + ex)
    assert mp.almosteq(p, expected_p)


@pytest.mark.parametrize('p', [0.125, 0.25, 0.5, 0.75, 0.875])
@mp.workdps(50)
def test_invcdf(p):
    p = mp.mpf(p)
    loc = mp.mpf(2)
    scale = mp.mpf(5)
    x = logistic.invcdf(p, loc=loc, scale=scale)
    expected_x = loc + scale*mp.log(p/(1 - p))
    assert mp.almosteq(x, expected_x)


@pytest.mark.parametrize('p', [0.125, 0.25, 0.5, 0.75, 0.875])
@mp.workdps(50)
def test_invsf(p):
    p = mp.mpf(p)
    loc = mp.mpf(2)
    scale = mp.mpf(5)
    x = logistic.invsf(p, loc=loc, scale=scale)
    expected_x = loc + scale*mp.log((1 - p)/p)
    assert mp.almosteq(x, expected_x)


@pytest.mark.parametrize('scale', [1, 0.5])
@mp.workdps(50)
def test_entropy(scale):
    ent = logistic.entropy(scale=scale)
    assert mp.almosteq(ent, mp.log(scale) + 2)


@mp.workdps(50)
def test_mle():
    # For the specially constructed data set in `data`, the MLE is
    # loc=3 and scale=0.5.
    alpha = mp.findroot(lambda t: t*mp.tanh(t) - 0.75, 1.0)
    loc = mp.mpf(3)
    data = [loc - alpha, loc, loc + alpha]
    loc1, scale1 = logistic.mle(data)
    assert mp.almosteq(loc1, loc)
    assert mp.almosteq(scale1, 0.5)
