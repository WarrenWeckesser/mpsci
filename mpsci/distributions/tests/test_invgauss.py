
import pytest
import mpmath as mp
from mpsci.distributions import invgauss


@pytest.mark.parametrize('mu', ['1e-6', 3])
def test_invgauss_entropy_against_integral(mu):

    def integrand(x):
        # This is a closure that captures mu.
        return invgauss.pdf(x, mu) * invgauss.logpdf(x, mu)

    with mp.workdps(50):
        mu = mp.mpf(mu)
        entr = invgauss.entropy(mu)
        with mp.extradps(25):
            m = invgauss.mode(mu)
            val = -mp.quad(integrand, [0, m, mp.inf])
        assert mp.almosteq(entr, val)
