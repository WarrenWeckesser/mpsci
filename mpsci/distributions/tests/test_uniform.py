import pytest
from mpmath import mp
from mpsci.distributions import uniform
from ._expect import check_noncentral_moment_with_integral


def test_validate_pdf():
    with pytest.raises(ValueError, match='must be less than'):
        uniform.pdf(3.5, 3, 2)


@pytest.mark.parametrize('x, ref', [(-3.0, 0.0), (2.0, 0.25), (6.0, 0.0)])
@mp.workdps(25)
def test_pdf(x, ref):
    a = 1.5
    b = 5.5
    p = uniform.pdf(x, a, b)
    assert p == ref


@pytest.mark.parametrize('x', [-3, 2, 6])
@mp.workdps(25)
def test_logpdf(x):
    a = 1.5
    b = 5.5
    p = uniform.logpdf(x, a, b)
    if x < a or x > b:
        assert p == mp.ninf
    else:
        assert mp.almosteq(p, mp.log(0.25))


@pytest.mark.parametrize('x', [-3, 2, 6])
@mp.workdps(25)
def test_cdf(x):
    a = 1.5
    b = 5.5
    p = uniform.cdf(x, a, b)
    if x < a:
        assert p == 0
    elif x > b:
        assert p == 1
    else:
        assert mp.almosteq(p, (x - a)/(b - a))


@pytest.mark.parametrize('x', [-3, 2, 6])
@mp.workdps(25)
def test_sf(x):
    a = 1.5
    b = 5.5
    p = uniform.sf(x, a, b)
    if x < a:
        assert p == 1
    elif x > b:
        assert p == 0
    else:
        assert mp.almosteq(p, (b - x)/(b - a))


@mp.workdps(20)
def test_invcdf():
    a = 3.0
    b = 11.0
    x = uniform.invcdf(0.25, a, b)
    assert x == 5.0


@mp.workdps(20)
def test_invsf():
    a = 3.0
    b = 11.0
    x = uniform.invsf(0.25, a, b)
    assert x == 9.0


@mp.workdps(25)
def test_mean():
    a = 4.0
    b = 6.0
    assert uniform.mean(a, b) == 5.0


@mp.workdps(25)
def test_median():
    a = 4.0
    b = 6.0
    assert uniform.median(a, b) == 5.0


@mp.workdps(25)
def test_var():
    a = 1.0
    b = 13.0
    # Variance is (b - a)**2/12
    # (https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    assert uniform.var(a, b) == 12


@mp.workdps(25)
def test_entropy():
    a = 1.0
    b = 13.0
    entr = uniform.entropy(a, b)
    # Differential entropy is ln(b - a)
    # (https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    assert mp.almosteq(entr, mp.log(b - a))


@pytest.mark.parametrize('order', [0, 1, 2, 3, 4])
@mp.workdps(50)
def test_noncentral_moment_with_integral2(order):
    a = -3.0
    b = 5.0
    check_noncentral_moment_with_integral(order, uniform, (a, b))


@mp.workdps(25)
def test_mle():
    x = [-2.5, -1.0, 0.5, 2.25, 3.0]
    a, b = uniform.mle(x)
    # MLE for the uniform distibution is (a, b) = (min(x), max(x))
    assert (a, b) == (min(x), max(x))
