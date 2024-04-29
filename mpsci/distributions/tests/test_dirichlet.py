
from mpmath import mp
from mpsci.distributions import dirichlet


def test_mean():
    # Expected result calculated "by hand"
    alpha = [0.5, 2.5, 5]  # Sum is 8
    mean = dirichlet.mean(alpha)
    for a, m in zip(alpha, mean):
        assert mp.almosteq(m, a/8)
