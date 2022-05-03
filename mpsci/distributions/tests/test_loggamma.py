
import mpmath
from mpsci.distributions import loggamma


def test_mean():
    with mpmath.workdps(50):
        k = mpmath.mpf(2)
        theta = mpmath.mpf(7)
        mean = loggamma.mean(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     Mean[ExpGammaDistribution[2, 7, 0]]
        valstr = '2.9594903456892699757544153694231829827048846484205348083596'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(mean, expected)


def test_var():
    with mpmath.workdps(50):
        k = mpmath.mpf(2)
        theta = mpmath.mpf(7)
        var = loggamma.var(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     Variance[ExpGammaDistribution[2, 7, 0]]
        valstr = '31.601769275563095387148343165655234271728545159133123449042'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(var, expected)
