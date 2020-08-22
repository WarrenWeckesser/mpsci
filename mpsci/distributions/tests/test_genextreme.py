
import mpmath
from mpsci.distributions import genextreme


def test_basic_pdf():
    with mpmath.workdps(50):
        # The expected value was computed "by hand".
        expected = mpmath.exp(mpmath.mpf('-0.5'))/16
        assert mpmath.almosteq(genextreme.pdf(6, 2, 3, 2), expected)


def test_basic_cdf():
    with mpmath.workdps(50):
        # The expected value was computed "by hand".
        expected = mpmath.exp('-0.5')
        assert mpmath.almosteq(genextreme.cdf(6, 2, 3, 2), expected)


def test_mean():
    with mpmath.workdps(50):
        xi = mpmath.mpf('0.5')
        mu = 3
        sigma = 2
        g1 = mpmath.gamma(1 - xi)
        assert mpmath.almosteq(genextreme.mean(xi, mu, sigma),
                               mu + sigma * (g1 - 1)/xi)


def test_mean_xi_zero():
    with mpmath.workdps(50):
        xi = 0
        mu = 3
        sigma = 2
        assert mpmath.almosteq(genextreme.mean(xi, mu, sigma),
                               mu + sigma * mpmath.euler)


def test_inf_mean():
    with mpmath.workdps(50):
        assert genextreme.mean(2, 3, 2) == mpmath.inf


def test_var():
    with mpmath.workdps(50):
        xi = mpmath.mpf('0.25')
        mu = 3
        sigma = 2
        g1 = mpmath.gamma(1 - xi)
        g2 = mpmath.gamma(1 - 2*xi)
        assert mpmath.almosteq(genextreme.var(xi, mu, sigma),
                               sigma**2 * (g2 - g1**2) / xi**2)


def test_var_xi_zero():
    with mpmath.workdps(50):
        xi = 0
        mu = 3
        sigma = 2
        assert mpmath.almosteq(genextreme.var(xi, mu, sigma),
                               sigma**2 * mpmath.pi**2 / 6)


def test_inf_var():
    with mpmath.workdps(50):
        assert genextreme.var(2, 3, 2) == mpmath.inf
