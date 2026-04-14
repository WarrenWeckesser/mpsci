import pytest
from mpmath import mp
from mpsci.distributions import benini
from ._expect import noncentral_moment_with_integral


@pytest.mark.parametrize('b', [1, 6])
def test_mode_endpoint(b):
    a = 3
    scale = 2.5
    m = benini.mode(a, b, scale)
    assert m == scale


@mp.workdps(50)
def test_mode_interior():
    # A crude test of the mode.
    a = 10
    b = 60  # b > a*(a + 1)/2
    scale = 1
    m = benini.mode(a, b, scale)
    pm = benini.pdf(m, a, b, scale)
    delta = mp.sqrt(mp.eps)
    assert benini.pdf(m - delta, a, b, scale) < pm
    assert benini.pdf(m + delta, a, b, scale) < pm


@pytest.mark.parametrize('alpha, beta, scale',
                         [(0.125, 25, 1),
                          (0.125, 25, 3),
                          (10, 0.25, 0.25),
                          (100, 80, 1)])
@mp.workdps(50)
def test_mean_with_integral(alpha, beta, scale):
    m = benini.mean(alpha, beta, scale)
    q = noncentral_moment_with_integral(1, benini, (alpha, beta, scale))
    assert mp.almosteq(m, q)


@mp.workdps(50)
def test_var_with_integral():
    alpha = 0.125
    beta = 3.0
    scale = 2.0
    mu = benini.mean(alpha, beta, scale)
    var = benini.var(alpha, beta, scale)
    expected = mp.quad(lambda t: (t - mu)**2*benini.pdf(t, alpha, beta, scale),
                       [scale, mp.inf])
    assert mp.almosteq(var, expected)
