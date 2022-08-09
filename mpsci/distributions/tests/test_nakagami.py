
from itertools import product
import pytest
import mpmath
from mpsci.distributions import nakagami


mpmath.mp.dps = 40


@pytest.mark.parametrize(
    'x',
    [[2, 4, 8, 16],
     [5.375, 4.625, 4.250, 5.125, 5.000, 5.125, 4.250, 4.500, 5.125, 5.500]]
)
def test_mle(x):
    # This is a crude test of nakagami.mle().
    nu_hat, _, scale_hat = nakagami.mle(x, loc=0)
    nll = nakagami.nll(x, nu=nu_hat, loc=0, scale=scale_hat)
    delta = 1e-9
    n = 2
    dirs = set(product(*([[-1, 0, 1]]*n))) - set([(0,)*n])
    for d in dirs:
        nu = nu_hat + d[0]*delta
        scale = scale_hat + d[1]*delta
        assert nll < nakagami.nll(x, nu=nu, loc=0, scale=scale)
