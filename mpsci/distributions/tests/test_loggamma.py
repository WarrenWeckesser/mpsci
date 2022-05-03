
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


def test_skewness():
    with mpmath.workdps(50):
        k = mpmath.mpf(2)
        theta = mpmath.mpf(7)
        skew = loggamma.skewness(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     Skewness[ExpGammaDistribution[2, 7, 0]]
        valstr = '-0.780244491437787061530103769675602605070996940757461555988'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(skew, expected)


def test_kurtosis():
    with mpmath.workdps(50):
        k = mpmath.mpf(2)
        theta = mpmath.mpf(7)
        kurt = loggamma.kurtosis(k, theta)
        # Expected value computed with Wolfram Alpha:
        #     ExcessKurtosis[ExpGammaDistribution[2, 7, 0]]
        valstr = '1.1875257511965620472368652875718220772212082979314426565409'
        expected = mpmath.mpf(valstr)
        assert mpmath.almosteq(kurt, expected)
