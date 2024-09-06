import pytest
from mpmath import mp
from mpsci.distributions import dirichlet


def test_bad_lengths():
    alpha = [0.25, 1.5, 1]
    with pytest.raises(ValueError, match='must be the same length'):
        dirichlet.pdf([0.25, 0.25, 0.25, 0.25], alpha)


def test_bad_alpha():
    alpha = [0.25, 1.5, -0.3, 1]
    with pytest.raises(ValueError, match='alpha must be positive'):
        dirichlet.pdf([0.25, 0.25, 0.25, 0.25], alpha)


def test_bad_x_value():
    alpha = [0.25, 1.5, 3.5, 1]
    with pytest.raises(ValueError, match=r'x must be nonnegative'):
        dirichlet.pdf([0.25, 0.25, 0.75, -0.25], alpha)


def test_bad_x_sum():
    alpha = [0.25, 1.5, 3.5, 1]
    with pytest.raises(ValueError, match=r'sum\(x\) must be 1'):
        dirichlet.pdf([0.25, 0.25, 0.25, 0.24], alpha)


@mp.extradps(10)
def f(x, alpha):
    x = mp.mpf(x)

    def inner(y):
        return dirichlet.pdf([x, y, 1 - (x + y)], alpha)

    return mp.quad(inner, [0, 1 - x])


@mp.workdps(35)
def test_integal_over_support():
    # Integral of the PDF over the support must be 1.
    alpha = [1.125, 1.25, 2.5]
    integral = mp.quad(lambda x: f(x, alpha), [0, 1])
    assert mp.almosteq(integral, 1)


def test_mean():
    # Expected result calculated "by hand"
    alpha = [0.5, 2.5, 5]  # Sum is 8
    mean = dirichlet.mean(alpha)
    for a, m in zip(alpha, mean):
        assert mp.almosteq(m, a/8)


@mp.extradps(10)
def entropy_slice(x, alpha):
    x = mp.mpf(x)

    def inner(y):
        p = dirichlet.pdf([x, y, 1 - (x + y)], alpha)
        logp = dirichlet.logpdf([x, y, 1 - (x + y)], alpha)
        return -p*logp if logp > mp.ninf else 0

    return mp.quad(inner, [0, 1 - x])


def entropy_integral(alpha):
    integral = mp.quad(lambda x: entropy_slice(x, alpha), [0, 1])
    return integral


@mp.workdps(35)
def test_entropy_against_integral():
    alpha = [1.125, 1.25, 2.5]
    ent = dirichlet.entropy(alpha)
    integral = entropy_integral(alpha)
    assert mp.almosteq(ent, integral)


@mp.extradps(15)
def cov01_slice(x, alpha):
    x = mp.mpf(x)
    s = sum(alpha)
    xbar = alpha[0]/s
    ybar = alpha[1]/s

    def inner(y):
        p = dirichlet.pdf([x, y, 1 - (x + y)], alpha)
        return (x - xbar)*(y - ybar)*p

    return mp.quad(inner, [0, 1 - x])


def cov01_integral(alpha):
    integral = mp.quad(lambda x: cov01_slice(x, alpha), [0, 1])
    return integral


@mp.workdps(40)
def test_cov01_against_integral():
    alpha = [mp.mpf('1.125'), mp.mpf('1.25'), mp.mpf('2.5')]
    cov = dirichlet.cov(alpha)
    integral01 = cov01_integral(alpha)
    assert mp.almosteq(cov[0, 1], integral01)
