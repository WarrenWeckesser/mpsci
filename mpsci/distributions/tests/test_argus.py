
import pytest
from mpmath import mp
from mpsci.distributions import argus
from ._expect import noncentral_moment_with_integral


@pytest.mark.parametrize('c', [-3.0, 0.0])
def test_validate_c(c):
    with pytest.raises(ValueError, match='c must be positive'):
        argus._validate_params(4.5, c)


def test_validate_chi():
    with pytest.raises(ValueError, match='chi must be nonnegative'):
        argus._validate_params(-1.5, 2)


def test_support():
    s = argus.support(1.0, 0.5)
    # The support is just (0, c).
    assert s == (mp.zero, mp.mpf(0.5))


@mp.workdps(25)
def test_pdf_outside_support():
    p = argus.pdf(-0.5, 5, 1)
    assert p == mp.zero


@mp.workdps(25)
def test_logpdf_outside_support():
    p = argus.logpdf(-2.5, 2, 0.5)
    assert p == mp.ninf


@mp.workdps(25)
def test_cdf_outside_support():
    p = argus.cdf(-0.5, 5, 1)
    assert p == mp.zero
    p = argus.cdf(1.5, 5, 1)
    assert p == mp.one


@mp.workdps(25)
def test_sf_outside_support():
    p = argus.sf(-0.5, 5, 1)
    assert p == mp.one
    p = argus.sf(1.5, 5, 1)
    assert p == mp.zero


# We don't have an external, independent source of truth for
# the ARGUS distribution, so we do a bunch of consistency checks.

@mp.workdps(50)
def test_pdf_integral_is_one():
    c = 0.5
    i = mp.quad(lambda t: argus.pdf(t, 1.0, c), [0, c])
    assert mp.almosteq(i, mp.one)


@pytest.mark.parametrize('x, chi, c', [(0.25, 3, 0.5), (0.95, 2.5, 1.0)])
@mp.workdps(50)
def test_pdf_integrates_to_cdf(x, chi, c):
    p = argus.cdf(x, chi, c)
    q = mp.quad(lambda t: argus.pdf(t, chi, c), [0, x])
    assert mp.almosteq(p, q), f'{p=} is not close to {q=}'


@pytest.mark.parametrize('x, chi, c', [(0.25, 3, 0.5), (0.95, 2.5, 1.0)])
@mp.workdps(50)
def test_pdf_integrates_to_sf(x, chi, c):
    p = argus.sf(x, chi, c)
    q = mp.quad(lambda t: argus.pdf(t, chi, c), [x, c])
    assert mp.almosteq(p, q), f'{p=} is not close to {q=}'


@pytest.mark.parametrize('chi, c',
                         [(0.5, 3.0),
                          (0.125, 25.0),
                          (10, 4),
                          (240, 725)])
@mp.workdps(50)
def test_mean_with_integral(chi, c):
    m = argus.mean(chi, c)
    q = noncentral_moment_with_integral(1, argus, (chi, c))
    assert mp.almosteq(m, q)


@mp.workdps(50)
def test_var_with_integral():
    chi = 3.5
    c = 1.25
    var = argus.var(chi, c)

    mean = argus.mean(chi, c)
    mom2 = noncentral_moment_with_integral(2, argus, (chi, c))
    assert mp.almosteq(var, mom2 - mean**2)


@mp.workdps(50)
def test_mode():
    # A crude test of the mode.
    chi = 4.0
    c = 0.125
    m = argus.mode(chi, c)
    pm = argus.pdf(m, chi, c)
    delta = mp.sqrt(mp.eps)
    assert argus.pdf(m - delta, chi, c) < pm
    assert argus.pdf(m + delta, chi, c) < pm
