
from mpmath import mp
from mpsci.distributions import genextreme


def test_basic_pdf():
    with mp.workdps(50):
        # The expected value was computed "by hand".
        expected = mp.exp(mp.mpf('-0.5'))/16
        assert mp.almosteq(genextreme.pdf(6, 2, 3, 2), expected)


def test_basic_cdf():
    with mp.workdps(50):
        # The expected value was computed "by hand".
        expected = mp.exp('-0.5')
        assert mp.almosteq(genextreme.cdf(6, 2, 3, 2), expected)


def test_mean():
    with mp.workdps(50):
        xi = mp.mpf('0.5')
        mu = 3
        sigma = 2
        g1 = mp.gamma(1 - xi)
        assert mp.almosteq(genextreme.mean(xi, mu, sigma),
                           mu + sigma * (g1 - 1)/xi)


def test_mean_xi_zero():
    with mp.workdps(50):
        xi = 0
        mu = 3
        sigma = 2
        assert mp.almosteq(genextreme.mean(xi, mu, sigma),
                           mu + sigma * mp.euler)


def test_inf_mean():
    with mp.workdps(50):
        assert genextreme.mean(2, 3, 2) == mp.inf


def test_var():
    with mp.workdps(50):
        xi = mp.mpf('0.25')
        mu = 3
        sigma = 2
        g1 = mp.gamma(1 - xi)
        g2 = mp.gamma(1 - 2*xi)
        assert mp.almosteq(genextreme.var(xi, mu, sigma),
                           sigma**2 * (g2 - g1**2) / xi**2)


def test_var_xi_zero():
    with mp.workdps(50):
        xi = 0
        mu = 3
        sigma = 2
        assert mp.almosteq(genextreme.var(xi, mu, sigma),
                           sigma**2 * mp.pi**2 / 6)


def test_inf_var():
    with mp.workdps(50):
        assert genextreme.var(2, 3, 2) == mp.inf
